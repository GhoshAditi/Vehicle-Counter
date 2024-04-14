import cv2
import numpy as np

# Web Camera
cap = cv2.VideoCapture('Smart-Traffic-Management/video.mp4')
min_width_react = 80   #min width rectangle
min_height_react = 80  #min height rectangle
count_line_position = 550

# Initialize Subtractor
#this algo will only detect the vehicle and the surrounding of the vehicle will not be detected
algo = cv2.bgsegm.createBackgroundSubtractorMOG()



def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6 # allowable error between pixel
counter=0

while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # Apply Subtractor on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    # Add erosion and dilation
    eroded = cv2.erode(dilatada, None, iterations=2)
    dilated = cv2.dilate(eroded, None, iterations=2)

    counterShape,h= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    # cv2.imshow('Detector', dilatada)

    for(i,c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
         # Filter based on aspect ratio to remove non-vehicle contours
        aspect_ratio = float(w)/h
        if aspect_ratio < 1 or aspect_ratio > 10:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, 'VEHICLE:'+str(counter), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)    

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
               counter+=1
               cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
               detect.remove((x, y))
               print("Vehicle Detected: "+str(counter))

    cv2.putText(frame1, 'VEHICLE COUNT: '+str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow('Video Original', frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
