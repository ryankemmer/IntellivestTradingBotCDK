FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.7.16-slim

RUN mkdir -p /var/task
WORKDIR /var/task

COPY requirements.txt .
RUN  pip3 install -r requirements.txt 

COPY app.py .

CMD [ "python", "app.py" ]