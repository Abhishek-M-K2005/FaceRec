import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime 
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")

with open('data/names.pkl') as f:
    Y = pickle.load(f)
    
with open("data/face_data.pkl") as f:
    X = pickle.load(f)    

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, Y)

COL_NAMES =["Name", "Time"]


while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crp_img = frame[y : y+h, x : x + w, :]
        rsz_img = cv2.resize(crp_img, (50, 50))
        output = knn.predict(rsz_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        ts = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        
        cv2.putText(frame, str(output), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y - 40), (0, 0, 255), 5)
        cv2.rectangle(frame, (x, y), (x+ w, y + h),(50, 50, 255), 5)
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()