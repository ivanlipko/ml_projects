##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Sat Dec  8 09:36:13 2018
#
#@author: ivan

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 09:36:13 2018

@author: ivan
"""

# second video

import cv2 as cv

def nothing(x):
    pass

cap = cv.VideoCapture(0)
cv.namedWindow('sliders')
cv.resizeWindow('sliders', 400,400)
cv.createTrackbar('min_b','sliders', 0, 255, nothing)
cv.createTrackbar('min_g','sliders', 0, 255, nothing)
cv.createTrackbar('min_r','sliders', 0, 255, nothing)
cv.createTrackbar('max_b','sliders', 255, 255, nothing)
cv.createTrackbar('max_g','sliders', 255, 255, nothing)
cv.createTrackbar('max_r','sliders', 255, 255, nothing)

imgNoDrive = cv.imread('./data/standards/no_drive.png')
imgPedestrian = cv.imread('./data/standards/pedistrain.jpg')
#imgNoDrive = cv.resize(imgNoDrive, (32,32))
#imgPedestrian = cv.resize(imgPedestrian, (32,32))
mask = cv.inRange(hsv,(50,40,60),(max_b,max_g,max_r))

while (True):
#    ret, frame = cap.read()
#    frameCopy = frame.copy(q)
    frame = imgNoDrive
    cv.imshow('source', frame)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (3,3))

    min_b = cv.getTrackbarPos('min_b','sliders')
    min_g = cv.getTrackbarPos('min_g','sliders')
    min_r = cv.getTrackbarPos('min_r','sliders')
    max_b = cv.getTrackbarPos('max_b','sliders')
    max_g = cv.getTrackbarPos('max_g','sliders')
    max_r = cv.getTrackbarPos('max_r','sliders')

#     Detect on mask
    mask = cv.inRange(hsv,(min_b,min_g,min_r),(max_b,max_g,max_r))
    mask = cv.erode(mask, (5,5), iterations=2)      # fill neighbours by black
#    cv.imshow('erode',mask)
    mask = cv.dilate(mask, (5,5), iterations=2)     # fill neighbours by white
    cv.imshow('dilate',mask)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#    contours is structure
    contours = contours[1] # array of contours

    if contours:        # array is not empty
        contours = sorted(contours, key=cv.contourArea, reverse=True)
#    Draw on frame
        (x,y,w,h) = cv.boundingRect(contours[0])
#        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        win_sliders = cv.bitwise_and(frame, frame, mask = mask)
        cv.imshow('sliders', win_sliders)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
# second video end




#"""
#
## first video
#
#import cv2 as cv
#
#def nothing(x):
#    pass
#
#cap = cv.VideoCapture(0)
#cv.namedWindow('sliders')
#cv.createTrackbar('min_b','sliders', 0, 255, nothing)
#cv.createTrackbar('min_g','sliders', 0, 255, nothing)
#cv.createTrackbar('min_r','sliders', 0, 255, nothing)
#cv.createTrackbar('max_b','sliders', 255, 255, nothing)
#cv.createTrackbar('max_g','sliders', 255, 255, nothing)
#cv.createTrackbar('max_r','sliders', 255, 255, nothing)
#
#while (True):
#    ret, frame = cap.read()
##    frameCopy = frame.copy(q)
#    cv.imshow('source', frame)
#    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#
#    min_b = cv.getTrackbarPos('min_b','sliders')
#    min_g = cv.getTrackbarPos('min_g','sliders')
#    min_r = cv.getTrackbarPos('min_r','sliders')
#    max_b = cv.getTrackbarPos('max_b','sliders')
#    max_g = cv.getTrackbarPos('max_g','sliders')
#    max_r = cv.getTrackbarPos('max_r','sliders')
#
##     Detect on mask
#    mask = cv.inRange(hsv,(min_b,min_g,min_r),(max_b,max_g,max_r))
##    cv.imshow('Mask', mask)
#    mask = cv.blur(mask, (5,5))
##    cv.imshow('blur',mask)
#    mask = cv.erode(mask, (5,5), iterations=5)      # fill neighbours by black
##    cv.imshow('erode',mask)
#    mask = cv.dilate(mask, (5,5), iterations=2)     # fill neighbours by white
#    cv.imshow('dilate',mask)
#    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
##    contours is structure
#    contours = contours[1] # array of contours
#
#    if contours:        # array is not empty
#        contours = sorted(contours, key=cv.contourArea, reverse=True)
##    Draw on frame
#        cv.drawContours(frame, contours, 0, (255, 0, 255), 2)
#        (x,y,w,h) = cv.boundingRect(contours[0])
#        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
#        win_sliders = cv.bitwise_and(frame, frame, mask = mask)
#        cv.imshow('sliders', win_sliders)
#
##        cv.imshow('Rectangle', frame)
#        roImg = frame[y:y+h, x:x+w]
#        cv.imshow('roImg', roImg)
#
##    print(help(cv.imshow))  # help
#
#    if cv.waitKey(1) == ord('q'):
#        break
#cap.release()
#cv.destroyAllWindows()
## first video end