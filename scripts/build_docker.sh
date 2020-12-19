#!/usr/bin/env sh

docker build --build-arg UID=`id -u` GID=`id -g` local:wslocalization .