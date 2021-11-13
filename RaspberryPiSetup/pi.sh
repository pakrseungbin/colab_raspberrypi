#!/bin/bash

###To configure Wifi and SSH if needed
#sudo raspi-config

sudo apt update
sudo apt install git vim python3-pip libwebp-dev libtiff5 libopencv-dev libatlas-base-bev libopenblas-dev libgtk2.0-dev -y
pyhton3 -m pip install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
python3 -m pip install torchvision-0.8.0a0+45f960c-cp37-cp37m-linux_armv7l.whl
python3 -m pip install opencv-python-headless=4.4.0.40
sed -i 's/torchvision>=0.8.1/#torchvision>=0.8.1/g' requirements.txt
sed -i 's/opencv-python>=4.1.2/#opencv-python>=4.1.2/g' requirements.txt
python3 -m pip install -r requirements.txt

###To start the software
#python3 detect.py --weights YOLOv5-Lite-best.pt --source 0