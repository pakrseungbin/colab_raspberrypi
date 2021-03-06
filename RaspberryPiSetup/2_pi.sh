#!/bin/bash

sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt install git vim python3-pip libwebp-dev libtiff5 libopencv-dev libatlas-base-dev libopenblas-dev libgtk2.0-dev -y
pip3 install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
pip3 install torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
pip3 install -r requirements.txt

