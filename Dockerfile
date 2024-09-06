FROM python:3.11-slim

WORKDIR /usr/src/app
# 将requirements.txt文件复制到工作目录
COPY requirements.txt .

# 安装requirements.txt中指定的所有依赖项
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .
EXPOSE 7860

# 添加环境变量，但不设置具体值
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV DEEPSEEK_API_KEY=""

CMD ["python", "prototype.py"]

# 运行时注意按照如下的方式运行
# docker run --env-file .env -p 7860:7860 your-image-name