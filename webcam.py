#!/usr/bin/python
# -*- coding:utf-8 -*-
# Import packages
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
from PIL import Image
import torch
# 이미지 경로를 받아오면 딥러닝 형태로 변환해주는 함수
import PIL.Image
import time
import torch.nn.functional as F
from datetime import datetime
# 이미지 경로를 받아오면 딥러닝 형태로 변환해주는 함수
def process_image(img):
        
    # # 변화된 사이즈
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    
    # Chnnel이 먼저되게끔 설정(tensor로 변환)
    img = img.transpose((2, 0, 1))
    
    # 0~1로 변환
    img = img/255
    
    # Normalize 실시
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # 배치사이즈삽입 1
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image

def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = F.softmax(output, dim=1)
    
    # 확률과 클래스를 반환
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

class VideoStream:
    
    def __init__(self,resolution=(640,480),framerate=30):
        # Pi Camera 초기화 
        self.stream = cv2.VideoCapture(0) #카메라 번호는 0
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	    # Variable to control when the camera is stopped
        self.stopped = False
        


    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
                return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False
    
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .pth file is located in',
                    default='model.pth')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    LABELMAP_NAME = args.labels
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    # Get path to current working directory
    CWD_PATH = os.getcwd()
    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)
    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME)
    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    for label in labels:
        os.makedirs(os.path.join(os.getcwd(), label), exist_ok=True)
    
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    print("Load Model")
    model = torch.load('./model.pth', map_location=device)
    model.eval()
    device = torch.device(device)
    model.to(device)
    print("Loaded Model!")
    # Initialize video stream
    #videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    vc = cv2.VideoCapture(0)
    time.sleep(1)
    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
        
        # Grab frame from video stream
        # frame1 = videostream.read()
        isv, frame = vc.read()
        if isv == None:
            continue
        # Acquire frame and resize to expected shape [1xHxWx3]
        # frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = process_image(frame_rgb)
        
        frame_resized = torch.Tensor(frame_resized).to(device)
        top_prob, top_class = predict(frame_resized, model)
        # print(top_prob, top_class)

        # 결과출력
        # print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class)
        # Draw framerate in corner of frame
        cv2.putText(frame,f'FPS: {frame_rate_calc:.2f}',(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(frame,f'{labels[top_class]}',(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Image Class', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # q키를 누르면 종료 'q', 스페이스바를 누르면 해당 사진에서 분류결과를 폴더에 저장!
        if cv2.waitKey(10) == ord('q'):
            break
        if cv2.waitKey(10) == ord(' '):
            
            full_path = os.path.join(os.getcwd(), labels[top_class],\
                                    f"{labels[top_class]}_{top_prob:.3f}_{datetime.now():'%Y_%m_%d_%H_%M_%S'}.jpg")
            imwrite(full_path, frame)
    # Clean up
    cv2.destroyAllWindows()
