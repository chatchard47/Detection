import cv2
import numpy as np
import imutils
import argparse
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import RPi.GPIO as GPIO
import time


def nothing(x):
    pass
def midpoint(ptA, ptB):
            return  ((ptA[0] + ptB[0]) * 0.5,(ptA[1] + ptB[1]) * 0.5)
cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)
cv2.namedWindow("Trackbars")
GPIO.setmode(GPIO.BCM)
GPIO.setup(26,GPIO.OUT)


cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - s", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - v", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - h", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - s", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - v", "Trackbars", 255, 255, nothing)

                   


while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - s", "Trackbars")
    l_v = cv2.getTrackbarPos("L - v", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    
    black_min = np.array([0 ,0, 0])
    black_max = np.array([50 ,50, 50])
    black = cv2.inRange(frame, black_min, black_max)
    opening = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)

    _, contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 1*cv2.arcLength(cnt, True), True)
        cv2.drawContours(opening, [approx], 0, (0, 0, 0), 5)

        #print(len(approx))
        if (len(approx)) == 1 :
            time.sleep(.5)
            GPIO.output(26,GPIO.LOW)
            print("OFF 3 !!:")
            time.sleep(.5)
            GPIO.output(26,GPIO.HIGH)
            print("ON 3 !!:")
            
        

    

    
    rangomax = np.array([l_h,l_s,l_v])
    rangomin = np.array([u_h,u_s,u_v])
    mascara = cv2.inRange(frame, rangomin, rangomax)
    opening = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    
    x,y,w,h = cv2.boundingRect (opening)
    cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0),4)
    cv2.circle(frame, ((x+w),(y+h)),6,(0,0,100),-1)

    edged = cv2.Canny(frame, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    #(cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric  = None
    

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
                
        orig = frame.copy()    
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
            
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (255, 255, 0), 2)

    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


            
    (tl, tr, br, bl) = box
        
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(0, 0, 255), 2)
    cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(0, 0, 255), 2)



    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric  is None:
        pixelsPerMetric = dB 
        
        dimA = (dA * 37) / 255
        dimB = (dB * 37) / 255

        points = dimA + dimB
        #print(points)
        
        #if  (points) > 31 :
            #time.sleep(.5)
            #GPIO.output(26,GPIO.LOW)
            #print("OFF 1 !!:")
            #time.sleep(.5)
            #GPIO.output(26,GPIO.HIGH)
            #print("ON 1 !!:")
        #if (points) > 5 or (points) <= 30 :
            #GPIO.output(26,GPIO.HIGH)
            #print("OFF 2 !!:")
        
                        
            

    cv2.putText(frame, "{:.2f}mm.".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(frame, "{:.2f}mm.".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

    
    
    cv2.imshow('camara',frame)
    #cv2.imshow('mask',mascara)
    cv2.imshow('black',opening)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
