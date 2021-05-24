#!/usr/bin/env bash
source conf.sh

if [[ -z $IMAGE_TAG ]]
then
  echo "No image tag provided. Not building image."
else
  docker build -t CoRL/tfdepth:$IMAGE_TAG .
fi