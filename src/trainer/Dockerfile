FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.7.16-slim

RUN mkdir -p /var/task
WORKDIR /var/task

COPY requirements.txt  requirements.txt
RUN  pip3 install -r requirements.txt 

COPY . . 

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]