import uvicorn
from fastapi import FastAPI

from tts_infer import TTSInfer

tts_infer = TTSInfer()

app = FastAPI()

@app.get("/")
async def tts_endpoint(command: str = None,text: str = None, ):
    print(command , text )
    return tts_infer.handle(command, text)

if __name__ == "__main__":
    uvicorn.run(app, host=tts_infer.host, port=tts_infer.port, workers=tts_infer.workers)