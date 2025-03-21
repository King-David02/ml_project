FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN apt-get update && apt-get install unzip -y && pip install -r requirements.txt
CMD ["python3", "application.py"]