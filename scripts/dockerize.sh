#!/usr/bin/env bash

# Inject environment variable to application configs
# sed \
# -e "s/__VARIABLE_1__/'$VARIABLE_1'/" \
# -e "s/__VARIABLE_2__/'$VARIABLE_2'/" \
# ../src/config/env_build.cfg > ../src/config/env.cfg

cd src

docker login -u $DOCKER_USER -p $DOCKER_PASS

docker build --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -t seekintgdp/hackathon-jobsense-service:$DOCKER_IMAGE_TAG .
docker push seekintgdp/hackathon-jobsense-service:$DOCKER_IMAGE_TAG
