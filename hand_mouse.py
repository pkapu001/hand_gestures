# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:25:26 2019

@author: rajma
"""

from keras.models import load_model
import pickle
import numpy as np
import os
import cv2
from pynput.mouse import Button, Controller
import time
from time import sleep
mouse= Controller()

model = load_model('gesture_recog_93.h5')
with open('classes_95.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    classes = pickle.load(f)
    f.close()
classes = {val:key for (key, val) in classes.items()}


inputsize = (128,128)
os.system('clear')
font = cv2.FONT_HERSHEY_SIMPLEX
perc =lambda x: x*100
y1, y2 , dy = 350, 640, 640-350
x1, x2 , dx = 50, 380, 380-50
cam = cv2.VideoCapture(0)
ispressed = False
dt = 0
st = 0
rt = 0

while True:
    _, frame = cam.read()
    hsv_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
    corner = hsv_frame[53:-100 ,300:]
    corner0 = frame[53:-100 ,300:]

    corner = hsv_frame[53:-100, 300:]
    lower = np.array([0, 40, 50], dtype="uint8")
    upper = np.array([70, 255, 255], dtype="uint8")

    lower_p = np.array([165, 19, 190], dtype="uint8")
    upper_p = np.array([179, 70, 255], dtype="uint8")

    # cv2.imshow('hsv',hsv_frame)
    mask = cv2.inRange(corner, lower, upper)
    mask2 = cv2.inRange(corner, lower_p, upper_p)
    mask = cv2.bitwise_or(mask, mask2)

    th = cv2.GaussianBlur(mask, (9, 9), 9)
    _, th = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.rectangle(frame, (300, 53), (640, 380), (225, 255, 0), 2)
    temp = cv2.resize(th,(128,128))
    temp = np.expand_dims(temp, axis = 0)
    temp = np.expand_dims(temp, axis = -1)
    result = model.predict_proba(temp)
    gesture = classes[np.argmax(result)]
    # center = [int(len(mask[0]) / 2), int(len(mask)/ 2)]
    if gesture == 'rock' and not ispressed:
        mouse.press(Button.left)
        ispressed = True
        sleep(0.2)
    elif gesture == 'five' and ispressed:
        mouse.release(Button.left)
        ispressed = False
        sleep(0.2)
    elif gesture == 'one' and (time.time() -st) > 1.5 :
        mouse.click(Button.left,1)
        st = time.time()
        # sleep(2)
    elif gesture == 'two'and (time.time() -dt) > 1.5:
        mouse.click(Button.left, 2)
        dt = time.time()
        # sleep(2)
    elif gesture == 'three'and (time.time() -rt) > 1.5:
        mouse.click(Button.right, 1)
        rt = time.time()
        # sleep(2)
    elif gesture == 'swag':
        # f= np.asarray(mask)
        # th = cv2.cvtColor(th, cv2.COLOR_HSV2BGR)
        # th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)

        # th = cv2.pyrMeanShiftFiltering(mask,51,91)
        contours,_ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(corner0,contours,-1,(0,255,0),6)
        c = max(contours,key= cv2.contourArea)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.drawContours(corner0, contours, -1, (0, 255, 0), 3)
        th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        cv2.circle(th, (cX, cY), 7, (0, 0, 255), -1)
        # print(f'({cX},{cY})')
        handloc = [cX,cY]
        center = [int(len(mask[0])/2),int(len(mask)/2)]
        cv2.circle(th,(center[0],center[1]),5,(255,255,0),2)
        deviation = list(map(lambda x,y: x-y,center,handloc))
        deviation[1] = -deviation[1]
        if -10 < deviation[0] <10 :
            movex = 0
        else:
            movex = (deviation[0] / 30) ** 3

        if -10 < deviation[1]  < 10:
            movey = 0
        else:
            movey = (deviation[1]/30)**3
        # print(f'{handloc} - {center} = {deviation}  ==> ( {movex} , {movey} ) ')

        mouse.move(movex,movey)
        sleep(0.005)

    try:
        th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    except:
        pass
    cv2.putText(th, f'{gesture}', (50, 50), font, 1, (173, 0, 132), 2, cv2.LINE_AA)
    # cv2.circle(th,(center[0],center[1]),5,(255,255,0),2)
    # cv2.imshow('hand', mask)
    cv2.imshow('original', frame)
    cv2.imshow('hand', th)
    # cv2.imshow('corner', corner0)
    k = cv2.waitKey(1)& 0xff
    if k ==27:
        break

cam.release()
cv2.destroyAllWindows()