FROM python:3.11-slim

WORKDIR /usr/src/app
RUN pip install --no-cache-dir gradio,openai
COPY ./src .
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "prototype.py"]