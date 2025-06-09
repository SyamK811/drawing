import cv2
import HandTrackingModule as htm
import matplotlib.pyplot as plt
import numpy as np
import time

detector = htm.handDetector()

draw_color = (0,0,0)
brush_size = 20
eraser_size = 50
p_time = 0

image_canvas = np.zeros((720,1280,3),np.uint8)

video = cv2.VideoCapture(0)     #this code is used for webcam

while True:
    success,img=video.read()
    img = cv2.resize(img,(1280,720))
    cv2.rectangle(img,pt1=[0,0],pt2=[1280,110],color=[0,0,0],thickness=-1)
    cv2.rectangle(img,pt1=[15,15],pt2=[260,90],color=[0,0,255],thickness=-1)
    cv2.rectangle(img,pt1=[265,15],pt2=[510,90],color=[255,0,0],thickness=-1)
    cv2.rectangle(img,pt1=[515,15],pt2=[760,90],color=[0,255,0],thickness=-1)
    cv2.rectangle(img,pt1=[765,15],pt2=[1010,90],color=[0,255,255],thickness=-1)
    cv2.rectangle(img,pt1=[1015,15],pt2=[1265,90],color=[255,255,255],thickness=-1)
    cv2.putText(img,text='ERASER',org=[1067,65],fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.3,color=[0,0,10],thickness=2)

#2 Find hand landmarks

    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    # print(lmlist)
    if len(lmlist)!=0:
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        # print(x1,y1)

#3 find which finger is up

        fingers = detector.fingersUp()
        # print(fingers)

#4 if two finger is up - selection mode

        if fingers[1] and fingers[2]:
            xp,yp=0,0
            # print('Selection mode')
            if y1<170:
                if 15<x1<260:
                    draw_color = (0,0,225)
                if 265<x1<510:
                    draw_color = (255,0,0)
                if 515<x1<760:
                    draw_color = (0,255,0)
                if 765<x1<1010:
                    draw_color = (0,255,255)
                if 1015<x1<1265:
                    draw_color = (0,0,0)
            cv2.rectangle(img,(x1,y1),(x2,y2),draw_color,-3)

#5 if one finger is up - drawing mode

        if (fingers[1] and not fingers[2]):
            # print('Drawing Mode')

            if xp ==0 and yp==0:
                xp = x1
                yp = y1

            if draw_color == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),draw_color,thickness=eraser_size)
                cv2.line(image_canvas,(xp,yp),(x1,y1),draw_color,thickness=eraser_size)
            else:
                cv2.line(img,(xp,yp),(x1,y1),draw_color,thickness=brush_size)
                cv2.line(image_canvas,(xp,yp),(x1,y1),draw_color,thickness=brush_size)
            xp,yp=x1,y1

            cv2.circle(img,(x1,y1),10,draw_color,-3)

    img_grey = cv2.cvtColor(image_canvas,cv2.COLOR_BGR2GRAY)
    _,img_inv = cv2.threshold(img_grey,20,255,cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv,cv2.COLOR_GRAY2BGR)

    img=cv2.bitwise_and(img,img_inv)
    img=cv2.bitwise_or(img,image_canvas)

    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img,str(int(fps)),(50,250),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)

    img=cv2.addWeighted(img,1,image_canvas,0.5,0)

    cv2.imshow('video',img)
    # cv2.imshow('canvas',image_canvas)
    if cv2.waitKey(1) & 0XFF == 27:
        break


video.release
cv2.destroyAllWindows()
