import re
from io import BytesIO
import json
import ffmpeg

import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from fastapi.responses import StreamingResponse

from text import cleaned_text_to_sequence
from text.cleaner import clean_text

from _lib import cnhubert
from _lib.mel_processing import spectrogram_torch
from _lib.models import SynthesizerTrn

from AR.models.t2s_lightning_module import Text2SemanticLightningModule


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)



class TTSInfer:
    def __init__(self,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_config("data/config.json")
        self.load_model()

        self.n_semantic = 1024
        self.splits = {"，","。","？","！",",",".","?","!","~",":","：","—","…",}

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config_base = config['base']
        for key, value in config_base.items():
            setattr(self, key, value)
        
        config_model = config[config_base['use_model']]
        for key, value in config_model.items():
            setattr(self, key, value)
        
    def load_model(self,):
        cnhubert_path = self.cnhubert_path
        bert_path = self.bert_path
        sovits_path = self.sovits_path
        gpt_path = self.gpt_path
        ref_wav_path = self.default_refer_path
        default_refer_path = self.default_refer_path
        prompt_text = self.default_refer_text

        cnhubert.cnhubert_base_path = cnhubert_path
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.tokenizer = tokenizer
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        bert_model = bert_model.to(self.device)
        self.bert_model = bert_model
        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]

        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"

        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        ssl_model = cnhubert.get_model()
        ssl_model = ssl_model.to(self.device)

        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers= hps.data.n_speakers,
            **hps.model)
        vq_model = vq_model.to(self.device)
        vq_model.eval()
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        hz = 50
        max_sec = config['data']['max_sec']
        t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])

        prompt_text = prompt_text.strip()
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k=torch.cat([wav16k,zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
        phones1, word2ph1, norm_text1 = clean_text(prompt_text)
        phones1 = cleaned_text_to_sequence(phones1)
        bert1 = self.get_bert_feature(norm_text1, word2ph1).to(self.device)

        refer = self.get_spepc(hps, default_refer_path)  # .to(self.device)
        refer = refer.to(self.device)

        self.hps = hps
        self.hz = hz
        self.max_sec = max_sec

        self.config = config

        self.prompt_semantic = prompt_semantic

        self.phones1 = phones1
        self.word2ph1 = word2ph1
        self.norm_text1 = norm_text1
        self.bert1 = bert1
        self.refer = refer
        self.zero_wav = zero_wav

        self.t2s_model = t2s_model
        self.vq_model = vq_model
        self.ssl_model = ssl_model
        
    def load_audio(self, file, sr):
        try: 
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        except Exception as e: 
            raise RuntimeError(f"Failed to load audio: {e}")
        return np.frombuffer(out, np.float32).flatten()

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)  #####输入是long不用管精度问题，精度随bert_model
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_spepc(self, hps, filename):
        audio = self.load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)
        return spec

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in self.splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def get_tts_wav(self, text):
        texts = text.strip().split("\n")
        audio_opt = []
        for text in texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            phones2, word2ph2, norm_text2 = clean_text(text)
            phones2 = cleaned_text_to_sequence(phones2)
            bert1 = self.get_bert_feature(self.norm_text1, self.word2ph1).to(self.device)
            bert2 = self.get_bert_feature(norm_text2, word2ph2).to(self.device)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(self.phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = self.prompt_semantic.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=self.config["inference"]["top_k"],
                    early_stop_num=self.hz * self.max_sec,
                )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次

            audio = (
                self.vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), self.refer
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )  ###试试重建不带上prompt部分
            audio_opt.append(audio)
            audio_opt.append(self.zero_wav)

        yield self.hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

    def handle(self, command, text): 
        if command == "ping":
            return 'ping'
        elif command != "tts":
            return 'false'
        with torch.no_grad():
            gen = self.get_tts_wav(text)
            sampling_rate, audio_data = next(gen)

        wav = BytesIO()
        sf.write(wav, audio_data, sampling_rate, format="wav")
        wav.seek(0)

        torch.cuda.empty_cache()
        return StreamingResponse(wav, media_type="audio/wav")



if __name__ == "__main__":
    tts_infer = TTSInfer()
    x = tts_infer.handle("tts", "你好，欢迎使用语音合成服务。")
    print(x)
