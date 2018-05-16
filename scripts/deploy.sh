#!/usr/bin/env bash

cd config/deploy

# replace environment variables
find . -type f -print0 | xargs -0 sed -i "s#__STAGE__#$STAGE#g"
find . -type f -print0 | xargs -0 sed -i "s#__APP_CONTAINER_MEMORY__#$APP_CONTAINER_MEMORY#g"
find . -type f -print0 | xargs -0 sed -i "s#__AWS_S3_BUCKET_DOCKER_AUTH__#$AWS_S3_BUCKET_DOCKER_AUTH#g"
find . -type f -print0 | xargs -0 sed -i "s#__DOCKER_IMAGE_TAG__#$DOCKER_IMAGE_TAG#g"

# zip the needed files for elastic beanstalk and run eb_deployer
zip -r $BUILD_NUMBER.$STAGE.zip .
eb_deploy -p $BUILD_NUMBER.$STAGE.zip -e $STAGE -c eb-deployer.yml
