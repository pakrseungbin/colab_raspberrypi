#!/bin/bash
sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get -y install python3-dev

sudo apt install cmake build-essential pkg-config git -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev -y
sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 -y
sudo apt install libatlas-base-dev liblapacke-dev gfortran -y
sudo apt install libhdf5-dev libhdf5-103 -y

sudo apt install python3-dev python3-pip python3-numpy -y

pip3 install opencv-contrib-python==4.5.2.52