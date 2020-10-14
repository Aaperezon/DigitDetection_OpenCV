from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
import cv2
import json
from os import remove
from os import path

cam = cv2.VideoCapture(0)

camWidth = 160
camHeight = 120
cam.set(cv2.CAP_PROP_FRAME_WIDTH,camWidth)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,camHeight)
upScalingWidth = 640
upScalingHeight = 480
cv2.namedWindow('image')
#Digits for internal process
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}
#JSON CONFIGURATION
p = './Scripts/Configuration.json'
def GetHSV():
    global H_low, H_high, S_low, S_high, V_low, V_high
    with open(p) as config:
        config = json.load(config)
    H_low = int(config['H_low'])
    H_high = int(config['H_high'])
    S_low = int(config['S_low'])
    S_high = int(config['S_high'])
    V_low = int(config['V_low'])
    V_high = int(config['V_high'])
    print("HSV config from local file")

def SaveHSV():
    if path.exists(p):
        remove(p)
    data = {
        "H_low" : H_low,
        "H_high" : H_high,
        "S_low" : S_low,
        "S_high" : S_high,
        "V_low" : V_low,
        "V_high" : V_high
    }
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    f.close()
    if path.exists(p):
        print ('Updated...')
def callback(x):
    pass
    
if path.exists(p):
    GetHSV()
else:
    H_low = 0
    H_high = 180
    S_low = 0
    S_high = 255
    V_low = 0
    V_high = 255
# create trackbars for color change
cv2.createTrackbar('lowH','image',H_low,180,callback)
cv2.createTrackbar('highH','image',H_high,180,callback)

cv2.createTrackbar('lowS','image',S_low,255,callback)
cv2.createTrackbar('highS','image',S_high,255,callback)

cv2.createTrackbar('lowV','image',V_low,255,callback)
cv2.createTrackbar('highV','image',V_high,255,callback)


def TrackbarHSV():
    global H_low, H_high, S_low, S_high, V_low, V_high
    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    if(ilowH != H_low or ihighH != H_high or ilowS != S_low or ihighS != S_high or ilowV != V_low or ihighV != V_high):
        H_low = ilowH
        H_high = ihighH
        S_low = ilowS
        S_high = ihighS
        V_low = ilowV
        V_high = ihighV
        SaveHSV()
        

while(True):
    global edged
    ret, image = cam.read()
    image = cv2.resize(image, (upScalingWidth, upScalingHeight))
    kernal = np.ones((5,5),np.uint8)
    TrackbarHSV()
    #Convert image to HSV scale
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    #Selected color recognition
    minRange = np.array([H_low, S_low, V_low])
    maxRange = np.array([H_high, S_high, V_high])
    
    colorMask = cv2.inRange(hsvFrame, minRange, maxRange) 
    colorMask = cv2.dilate(colorMask, kernal) 
    res_color = cv2.bitwise_and(image, image, mask = colorMask) 
    

    
    gray = cv2.cvtColor(res_color, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    
    
    

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = []
    colorSquare  = cnts
    global posX,posY,posW,posH
    for c in colorSquare:
        if cv2.contourArea(c) > 2000:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if the contour has four vertices, then we have found
            # the thermostat display
            if len(approx) == 4:
                displayCnt = approx
                (posX, posY, posW, posH) = cv2.boundingRect(c)
                cv2.rectangle(image, (posX, posY), (posX + posW, posY + posH), (255, 0, 0), 3) #put contour in the original image
                break
        else:
            break
        
    
    if(len(displayCnt) == 4):
        # extract the thermostat display, apply a perspective transform
        # to it
        warped = four_point_transform(gray, displayCnt.reshape(4, 2))
        output = four_point_transform(image, displayCnt.reshape(4, 2))
        # threshold the warped image, then apply a series of morphological
        # operations to cleanup the thresholded image
        thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        
        
        
        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digitCnts = []
        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit
            if ((w >= 15) and (h >= 30 and h <= 200)):
                #print (str(w)+"   "+str(h))
                cv2.rectangle(image, (x+posX, y+posY), (x+posX + w, y+posY + h), (255, 0, 0), 3) #put contour in the original image
                digitCnts.append(c)
           
                
        #print ("Digitos: "+str(len(digitCnts)))
        if(len(digitCnts) >= 1 ):
            # sort the contours from left-to-right, then initialize the
            # actual digits themselves
            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
            digits = []
            #print (digitCnts)
        
            # loop over each of the digits
            for c in digitCnts:
                # extract the digit ROI
                (x, y, w, h) = cv2.boundingRect(c)
                #print (str(w)+"   "+str(h))
                #cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 0, 0), 3)
                roi = thresh[y:y + h, x:x + w]
                # compute the width and height of each of the 7 segments
                # we are going to examine
                (roiH, roiW) = roi.shape
                (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
                dHC = int(roiH * 0.05)
                # define the set of 7 segments
                segments = [
                    ((0, 0), (w, dH)),  # top
                    ((0, 0), (dW, h // 2)), # top-left
                    ((w - dW, 0), (w, h // 2)), # top-right
                    ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                    ((0, h // 2), (dW, h)), # bottom-left
                    ((w - dW, h // 2), (w, h)), # bottom-right
                    ((0, h - dH), (w, h))   # bottom
                ]
                on = [0] * len(segments)
                # loop over the segments
                for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                    # extract the segment ROI, count the total number of
                    # thresholded pixels in the segment, and then compute
                    # the area of the segment
                    segROI = roi[yA:yB, xA:xB]
                    total = cv2.countNonZero(segROI)
                    area = (xB - xA) * (yB - yA)
                    # if the total number of non-zero pixels is greater than
                    # 50% of the area, mark the segment as "on"
                    if total / float(area) > 0.5:
                        on[i]= 1
                # lookup the digit and draw it on the image
                if(tuple(on) in DIGITS_LOOKUP):
                    digit = DIGITS_LOOKUP[tuple(on)]
                    #print(str(digit)+"    "+ str(tuple(on)))
                    digits.append(digit)
                    #cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(image, str(digit), (posX+x - 10, posY+y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        #cv2.imshow('asd',thresh)
    cv2.imshow('image',res_color)
    cv2.imshow('result',image)
    k = cv2.waitKey(1)
    if(k==27):
        cv2.destroyAllWindows()
        pass
