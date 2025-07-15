#!/bin/bash -l

cp  -u /mnt/c/Users/damia/Downloads/cscs-key-cert.pub ~/.ssh
cp  -u /mnt/c/Users/damia/Downloads/cscs-key ~/.ssh

eval $(ssh-agent)
chmod 600 ~/.ssh/cscs-key
ssh-add ~/.ssh/cscs-key