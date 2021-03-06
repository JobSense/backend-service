import os
import sys
import threading

from boto3.s3.transfer import S3Transfer
import boto3
import botocore

from src.config.common import *


def main():
    """
    Simple script to download model from S3
    """
    download_model()


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            sys.stdout.write("\r%s --> %s bytes transferred" %
                             (self._filename, self._seen_so_far))
            sys.stdout.flush()


def download_dir(client, resource, dist, local, bucket):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get(
                    'Prefix'), local, bucket)
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                if not os.path.exists(os.path.dirname(local + os.sep + file.get('Key').replace(dist + '/', ""))):
                    os.makedirs(os.path.dirname(
                        local + os.sep + file.get('Key').replace(dist + '/', "")))
                resource.meta.client.download_file(
                    bucket, file.get('Key'), local + os.sep + file.get('Key').replace(dist + '/', ""))


def download_model():
    s3 = boto3.resource('s3')

    try:
        transfer = S3Transfer(boto3.client('s3', 'ap-southeast-1'))
        for key in KEYS:
            transfer.download_file(BUCKET_NAME, key, '/tmp/models/' + key.split(
                '/')[-1], callback=ProgressPercentage(key.split('/')[-1]))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


if __name__ == '__main__':
    main()
