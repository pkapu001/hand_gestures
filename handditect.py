import pickle
from time import sleep

import cv2
import numpy as np
import os


folders = ['one','two','three','four','five','rock','yo','swag','gun','none']

cam = cv2.VideoCapture(0)
# folder = input("name the gesture: ")
try:
    with open('endfilename.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        start_num = pickle.load(f)
except:
    start_num = 1
data_size = int(input("number of images for class : "))
test_size = int(data_size*0.10)
train_size = data_size - test_size
_, frame = cam.read()
sleep(2)
data = []
for folder in folders:
    temp = list(range(5))
    temp.reverse()
    for i in temp:
        os.system('clear')
        print(f'colecting data for {folder}  starts in {i+1} sec')
        sleep(1)


    i = start_num
    while True:
        _, frame = cam.read()
        # frame = cv2.flip(frame,+1)
        hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        corner = hsv_frame[53:-100 ,300:]
        lower = np.array([0, 40, 72], dtype="uint8")
        upper = np.array([70,255, 255], dtype="uint8")

        lower_p = np.array([165, 19, 190], dtype="uint8")
        upper_p = np.array([179, 70, 255], dtype="uint8")

        mask = cv2.inRange(corner,lower, upper)
        mask2 = cv2.inRange(corner, lower_p, upper_p)
        mask = cv2.bitwise_or(mask , mask2)
        th = cv2.GaussianBlur(mask, (9, 9), 11)
        _, th = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.circle(bgr,(145,165),10,(255,255,0),3)
        # cv2.imshow('hand1', bgr)

        cv2.imshow('full image', frame)
        cv2.imshow('hand', th)

        if i < start_num+ train_size:
            print(f'train img : {i}')
            cv2.imwrite(f'dataset/training/{folder}/{i}.jpg', th)
        else:
            cv2.imwrite(f'dataset/testset/{folder}/{i}.jpg', th)
            print(f'test img : {i}')

        i +=1
        key = cv2.waitKey(50)
        if key == 27:
            break
        elif key == 0:
            print(corner[165][145])
            x= corner[165][145]
            data.append(x)
        elif i > start_num + data_size:
            break

# data = np.asarray(data)
# print(f'\n===============================\n{data}\n\n==============================')
# print(f'min : {min(data[:,0])},{min(data[:,1])},{min(data[:,2])}    max : {max(data[:,0])},{max(data[:,1])},{max(data[:,2])}')
with open('endfilename.pkl', 'wb') as f:
    pickle.dump(start_num+data_size+1,f)

cam.release()
cv2.destroyAllWindows()
