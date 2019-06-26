import cv2
import numpy as np

cam = cv2.VideoCapture(0)
folder = input("name the gesture: ")
i = 1
while True:
    _, frame = cam.read()
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV_FULL)
    corner = hsv_frame[50:-100 ,350:]
    lower = np.array([0, 30, 50])
    upper = np.array([45, 255, 255])
    # cv2.imshow('hsv',hsv_frame)
    mask = cv2.inRange(corner,lower, upper)
    cv2.circle(corner,(145,165),10,(255,255,255),3)
    cv2.imshow('hsv2',frame[53:-100 ,350:])
    cv2.imshow('mask', mask)
    if 1< i <600:
        cv2.imwrite(f'dataset/training/{folder}/{i}.jpg', mask)
    else:
        cv2.imwrite(f'dataset/testset/{folder}/{i}.jpg', mask)
    print(f'img{i} is saved')
    i +=1
    key = cv2.waitKey(50)
    if key == 27:
        break
    elif key == 0:
        print(corner[165][145])
    elif i >650:
        break

cam.release()
cv2.destroyAllWindows()
