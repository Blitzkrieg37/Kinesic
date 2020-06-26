import numpy as np
import cv2
import time
from copy import deepcopy

def remove_face(frame,face_cascade):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        h = int(1.7*h)
        center = (x + w // 2, y + h // 2)
        w = int(1.1*w)
        frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (0, 0, 0), -1)

    return frame


def  extract_skin(frame, HSV_lower=(0, 14, 0), HSV_upper=(20,173,255), YCrCb_lower=(0,135,85), YCrCb_upper=(255,180,135), show=True):
    
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, HSV_lower, HSV_upper)
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
 
    YCrCb_mask = cv2.inRange(img_YCrCb, YCrCb_lower, YCrCb_upper) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)

    skin = cv2.bitwise_not(global_mask)

    if show:
        cv2.imshow("HSV skin",HSV_result)
        cv2.imshow("YCrCb skin",YCrCb_result)
        cv2.imshow("Global skin",skin)

    return skin


def extract_hand(frame,x,y,w,h,show = True):

    w,h = int(w*1.3),int(h*1.3)

    frame = deepcopy(frame[max(y-h//2,0):min(y+h//2,480),max(x-w//2,0):min(x+w//2,640)])
    frame = cv2.resize(frame,(120,200),cv2.INTER_CUBIC)

    kernel = np.ones((3,3),np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame = cv2.GaussianBlur(frame,(9,9),0)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((5,5),np.uint8)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    ret,frame = cv2.threshold(frame,60,255,cv2.THRESH_BINARY)

    if show:
        cv2.imshow("Segmented hand",frame)

    return frame


def remove_bg(fgbg,frame,show = True):
    fgmask = fgbg.apply(frame,learningRate=0)  #remove bg
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    frame = cv2.bitwise_and(frame,frame,mask=fgmask)

    if show:
        cv2.imshow("Backgroung subtracted",frame)

    return frame


def transform(frame,show=True):
    kernel = np.ones((5,5),np.uint8)
    frame = 255 - frame

    for i in range(2):
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        frame = cv2.threshold(frame,80,255,cv2.THRESH_BINARY)[1]
    if show:
        cv2.imshow("After first blur",frame)
    
    frame = cv2.dilate(frame,np.ones((3,3),np.uint8),iterations = 1)

    if show:
        cv2.imshow("After dilate", frame)

    for i in range(3):
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

    if show:
        cv2.imshow("After morphs",frame)
    
    blurValue = 31
    frame = cv2.medianBlur(frame,15) if 0 else cv2.threshold(cv2.GaussianBlur(frame, (blurValue, blurValue), 0),127,255,cv2.THRESH_BINARY)[1]

    if show:
        cv2.imshow("After last blur", frame)

    return 255 - frame


def get_centre(contour):
    cX, cY = None, None
    
    if contour is not None:
        M = cv2.moments(contour)
        _,_,w,h = cv2.boundingRect(contour)
        if M["m00"]:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

    return cX, cY

def get_gesture(skin,contour,cX,cY,model,show = True): 
    gesture = "None"
    
    if contour is not None and cX:
        _,_,w,h = cv2.boundingRect(contour)
        
        hand = extract_hand(255-skin,cX,cY,w,h)

        hand = np.expand_dims(hand,axis = -1)
        hand = np.expand_dims(hand,axis = 0)
        

        t = time.time()         
        
        probs = model.predict(hand)[0]
        pred = np.argmax(probs)

        if show:
            print("Forward pass:",time.time()-t)
            print("Predicted probs:",probs)

        gestures = ["Palm","Two","Fist"]
        if np.max(probs) > 0.5:
            gesture = gestures[pred]

    return gesture
