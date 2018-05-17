FROM ubuntu:16.04
LABEL maintainer="nicholedean, yimkh92@gmail.com"

RUN apt update -y

# Install java (for pyspark)
# RUN apt install default-jdk -y
# ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Install mysql
RUN DEBIAN_FRONTEND=noninteractive apt install mysql-client mysql-server libmysqlclient-dev -y -f

# Install python 2.7
RUN apt install python python-pip -y

# Add and install Python modules
ADD requirements.txt /python/src/requirements.txt
RUN cd /python/src; pip install -r requirements.txt

# Environment variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV APP_ENV=dev
ENV PYTHONPATH=/python
ENV APP_ENV_CONFIG=${PYTHONPATH}/src/config/env.cfg
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# Bundle app source
RUN mkdir /var/log/python
ADD . /python/src

# Download the model files to this docker image (host in local filesystem)
RUN mkdir -p /tmp/models
RUN mkdir -p /tmp/pipelines
RUN mkdir -p /tmp/inputs
RUN python /python/src/download_model.py

# Expose the port for API
EXPOSE 9000

# Run the API
WORKDIR /python/src
CMD ["gunicorn", "--bind", "0.0.0.0:9000", "--timeout", "180", "--workers", "1", "wsgi:app"]
