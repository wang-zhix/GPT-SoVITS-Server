# CPU镜像
# FROM cnstark/pytorch:2.0.0-py3.9.12-ubuntu20.04
# GPU镜像
FROM cnstark/pytorch:2.0.0-py3.9.12-cuda11.8.0-ubuntu20.04

# 设置工作目录
WORKDIR /home

# 复制当前目录下的所有文件到工作目录
COPY . /home

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=Etc/UTC

# 安装ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

# 安装可能需要的依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
