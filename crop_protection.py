# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Birds and Animal detection
import numpy as np
import cv2
import time
import requests
import pyglet
classNames = ['brid','cow', 'sheep']
COLORS = np.random.uniform(0, 255, size=(len(classNames), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5
value=0

# VideoCapture
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
j =0
while True:
    value=0
    
    _,image = cap.read()
    cv2.imshow("img",image)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (inWidth,inHeight)), inScaleFactor,
        (inWidth, inHeight), meanVal)
    net.setInput(blob)
    detections = net.forward()
    cols = image.shape[1]
    rows = image.shape[0]
    if cols / float(rows) > WHRatio:
        cropSize = (int(rows * WHRatio), rows)
    else:
        cropSize = (cols, int(cols / WHRatio))


    y1 = int((rows - cropSize[1]) / 2)
    y2 = y1 + cropSize[1]
    x1 = int((cols - cropSize[0]) / 2)
    x2 = x1 + cropSize[0]
    image = image[y1:y2, x1:x2]

    cols = image.shape[1]
    rows = image.shape[0]


    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 3 or class_id == 10 or class_id == 17 :
                if class_id == 3:
                    cattle = 0
                elif class_id ==10:
                    cattle = 1
                else:
                    cattle = 2
                    
                #print(class_id)
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                cv2.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))

                label = classNames[cattle] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                cv2.imwrite('./Animals/detected_'+str(j)+'.png', image)
                j +=1
                print ('animals detected')
                print(classNames[cattle]+" detected")
                value=1
                r= requests.get('https://di.eu-gb.mybluemix.net/data?animal='+str(classNames[cattle]))
                print(r.status_code)
               
     
    if value==1:
        print ('animal detected in field')
        music = pyglet.resource.media('alarm.mp3')
        music.play()
        pyglet.app.run()
        pyglet.app.exit()
       

        
       
    elif value==0:
        print('animal not detected')
    else:
        print ('')

    
    time.sleep(2) 

     #out.write(image)
    cv2.imshow('image',image)
    if cv2.waitKey(41) == 27:
        break

cap.release()
cv2.destroyAllWindows()
#out.release()



