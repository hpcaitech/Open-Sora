#!/usr/bin/env bash

# get root dir
FOLDER_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=$FOLDER_DIR/../..

# download at root dir
cd $ROOT_DIR
mkdir -p dataset && cd ./dataset
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip MSRVTT.zip