import cv2
import numpy as np
from copy import deepcopy
from keras.models import load_model
import time
import pyautogui as pg
from img_utils import *
pg.PAUSE = 0

def get_data(frame,fgbg, show = True):
    timf = time.time()
    frame = remove_face(frame, face_cascade)  #remove face
    print("Face removal:", time.time()-timf)
    ######################
    frame = remove_bg(fgbg,frame,show = show)  ## remove background 
    #######################
    skin = extract_skin(frame,show=show) ##extract skin
    ###########################################
    
    processed_skin = transform(skin,show=show)
    
    contours,_ = cv2.findContours(processed_skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ####################################################
    hand_contour = None
    w,h = None,None
    if len(contours):
        hand_contour = sorted(contours, key = cv2.contourArea, reverse=True)[1 if len(contours)>1 else 0]
        _,_,w,h = cv2.boundingRect(hand_contour)
        if show:
            print("Area",cv2.contourArea(hand_contour))
            cv2.drawContours(frame, [hand_contour], 0, (255,0,0), 3)
 
    cX,cY = get_centre(hand_contour)
    gesture = get_gesture(skin,hand_contour,cX,cY,model,show=show)
    
    if show:
        if cX:
            cv2.circle(frame, (cX, cY), 7, (255, 0, 0), -1)
        cv2.rectangle(frame,(70,110),(490,370),(255,0,0),2)
        frame = cv2.flip(frame,1)
        cv2.putText(frame, gesture,(200,470),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Frame",frame)

    return cX,cY,w,h,gesture


cap=cv2.VideoCapture(0)

##########################################
##########################################
gHSV_lower = None
gHSV_upper = None
gYCrCb_lower = None
gYCrCb_upper = None

while(True):
    ret,frame = cap.read()

    cv2.rectangle(frame,(300,300),(330,330),(255,0,0),2)
    cv2.imshow("Skin calibrator",frame)
    
    if cv2.waitKey(10) == ord('s'):
        frame = frame[304:325,304:325]

        frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

        H,S,V =  np.split(frame1,3,axis=-1)
        Y,Cr,Cb = np.split(frame2,3,axis=-1)

        gHSV_lower = np.array([np.min(H) - 30, np.min(S) - 80, 0])
        gHSV_upper = np.array([np.max(H) + 30, np.max(S) + 80, 255])

        gHSV_lower  = np.maximum(gHSV_lower,0)
        gHSV_upper = np.minimum(gHSV_upper,255)

        gYCrCb_lower = np.array([0, np.min(Cr) - 60, np.min(Cb) - 70])
        gYCrCb_upper = np.array([255, np.max(Cr) + 60, np.max(Cb) + 70])

        gYCrCb_upper = np.minimum(gHSV_upper,255)
        gYCrCb_lower = np.maximum(gHSV_lower,0)

        break

limits = (gHSV_lower,gHSV_upper,gYCrCb_lower,gYCrCb_upper) 
cv2.destroyAllWindows()

##########################################################
##########################################################

fgbg = None

while True:
    ret,frame = cap.read()
    cv2.imshow("Set background",frame)
    if cv2.waitKey(10) == ord('b'):
        fgbg = cv2.createBackgroundSubtractorMOG2(20,16,False)
        break
###########################################################

face_cascade = cv2.CascadeClassifier()
face_cascade.load('haar_face.xml')

###########################################################

model = load_model('models/gesture_2_classifier')

prev_gesture = "Palm"
ges_time = time.time()
done = 0
show = True 

while(True):
    ret,frame = cap.read()

    tim = time.time()
    cX,cY,w,h,gesture = get_data(frame,fgbg,show=show)
    print("Data extraction",time.time()-tim)
    

    if prev_gesture != gesture:
        ges_time = time.time()
        done = 0 

    if gesture == "Palm":
        pass  
        if cX and cY:
            pass
            px = 1919/420*(cX -70)
            py = 1079/260*(cY - 110)
            px = np.clip(px,1,1918)
            py = np.clip(py,1,1078)
            pg.moveTo(1919-px,py,0.00001)
        #pg.mouseDown(*pg.position(),button="left")
        #pg.click()
    if gesture == "Fist" and (time.time() - ges_time > 0.02) and not done:
        pass   
        pg.click(button="left")
        #pg.press('space')
        done = 1
    if gesture == "Two" and (time.time() - ges_time > 0.02) and not done:
        pass   
        #pg.press('down')
        pg.click(button = "right")
        done = 1

 
    prev_gesture = gesture
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 


