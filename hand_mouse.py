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
from hand_deviation_calc import *

mouse= Controller()

model = load_model('gesture_recog_81.h5')
with open('classes_81.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
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
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    corner = hsv_frame[50:-100 ,350:]
    corner0 = frame[50:-100 ,350:]
    lower = np.array([0, 40, 50],dtype = "uint8")
    upper = np.array([30, 255, 255],dtype = "uint8")
    mask = cv2.inRange(corner, lower, upper)
    th = cv2.GaussianBlur(mask, (21, 21), 41)
    cv2.rectangle(frame, (350, 50), (640, 380), (225, 255, 0), 2)
    temp = cv2.resize(mask,(128,128))
    temp = np.expand_dims(temp, axis = 0)
    temp = np.expand_dims(temp, axis = -1)
    result = model.predict_proba(temp)
    # print(list(map(perc , result)))
    gesture = classes[np.argmax(result)]

    if gesture == 'rock' and not ispressed:
        mouse.press(Button.left)
        ispressed = True
        sleep(0.2)
    elif gesture == 'five' and ispressed:
        mouse.release(Button.left)
        ispressed = False
        sleep(0.2)
    elif gesture == 'one' and (time.time() -st) > 3 :
        mouse.click(Button.left,1)
        st = time.time()
        # sleep(2)
    elif gesture == 'two'and (time.time() -dt) > 3:
        mouse.click(Button.left, 2)
        dt = time.time()
        # sleep(2)
    elif gesture == 'three'and (time.time() -rt) > 3:
        mouse.click(Button.right, 1)
        rt = time.time()
        # sleep(2)
    elif gesture == 'yo':
        # f= np.asarray(mask)
        # th = cv2.cvtColor(th, cv2.COLOR_HSV2BGR)
        # th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)
        # _, th = cv2.threshold(th,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # th = cv2.pyrMeanShiftFiltering(mask,51,91)
        contours,_ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(corner0,contours,-1,(0,255,0),6)
        c = max(contours,key= cv2.contourArea)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.drawContours(corner0, contours, -1, (0, 255, 0), 3)
        cv2.circle(corner0, (cX, cY), 7, (255, 255, 255), -1)
        # print(f'({cX},{cY})')
        handloc = [cX,cY]
        center = [(len(mask[0])/2),(len(mask)/2)+60]
        deviation = list(map(lambda x,y: x-y,center,handloc))
        deviation[1] = -deviation[1]
        print(f'{handloc} - {center} = {deviation}')
        mouse.move(deviation[0]/8,deviation[1]/5)
        sleep(0.005)

        # print(len(contours))


        # fx ,fy = f.sum(axis=0), f.sum(axis=1)
        # x, _ = find_weighted_center(fx)
        # y, _= find_weighted_center(fy)
        # cv2.circle(mask, (x, y), 10, (150, 200, 255), 3)
        # print(f'({x},{y})')
    cv2.putText(mask, f'{gesture}', (50, 50), font , 1, (173,0,132),2, cv2.LINE_AA)
    cv2.imshow('hand', mask)
    cv2.imshow('original', frame)
    cv2.imshow('blur', th)
    # cv2.imshow('corner', corner0)
    k = cv2.waitKey(1)& 0xff
    if k ==27:
        break

cam.release()
cv2.destroyAllWindows()