# RaspberryPiSetup
Repo for files needed to set up a Raspberry Pi with the PyTorch software

## Procedure for setting up the Pi
Simply download ```pi.sh``` and run it on your Pi with ```./pi.sh```. This will set up the environment and install all needed dependencies. This has been tested on a Raspberry Pi 3 Model B v1.2 with a fresh install of Raspbian 10 Buster

### Steps explained
Here is an explanation on what pi.sh does:
1. Run the ```raspi-config``` command and set up wifi and ssh if needed
2. Install python3.7 along with the matching version of pip
3. Install other required packages from the list with apt
4. Clone this repo in working directory and cd into it
5. Install both the torch and torchvision .whl files from this
6. Clone the latest Yolo-V5 repo containing the correct weight file in working directory
7. Comment out the torch and torchvision lines in requirements.txt and install it with pip
8. Cd into it, and run ```python3 detect.py --weights <weightfile.pt> --source <image.png, or 0 for camera>``` 

### Notes
- Raspberry Pi 3 Model B v1.2 uses armv7l archetecture, and official builds for the required version of torch and torchvision for armv7l do not exist. The whl files were compiled from source in a Docker container emulating an armv7l architecture using qemu. Trying to build from source directly on a Pi would take several days. See https://ownyourbits.com/2018/06/27/running-and-building-arm-docker-containers-in-x86/ for more information on this process.
- If you get an error of type ```filename.whl is not a supported wheel on this platform``` when installing torch or torchvision, check to make sure you have the correct python version instlalled (3.7 in this case). The default version of python3 differs in different OS releases.
