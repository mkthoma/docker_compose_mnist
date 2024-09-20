FROM python:3.9-slim

WORKDIR /opt

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /opt/mount
