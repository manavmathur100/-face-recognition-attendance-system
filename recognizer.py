import cv2, numpy as np;
import xlwrite,firebase_admin as fire;
import time
import sys
from playsound import playsound
start=time.time()
period=8
face_cas = cv2.CascadeClassifier('haarcascade.xml')
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainer.yml');
flag = 0;
id=0;
classes = ['class1', 'class2', 'class3']  # List of class names

for class_name in classes:
    filename = class_name + '.xlsx';  # Dynamic file name for each class
    dict = {};  # Dictionary to store attendance records for each class
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        id,conf=recognizer.predict(roi_gray)
        if(conf < 50):
         if(id==1):
            id='manav'
            if((str(id)) not in dict):
                filename=xlwrite.output('attendance','class1',1,id,'yes');
                dict[str(id)]=str(id);
                
         elif(id==2):
            id = 'raghav'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
                dict[str(id)] = str(id);

         elif(id==3):
            id = 'gk mishra'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 3, id, 'yes');
                dict[str(id)] = str(id);

         elif(id==4):
            id = 'ankit'
            if ((str(id)) not in dict):
                filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
                dict[str(id)] = str(id);

        else:
             id = 'Unknown, can not recognize'
             flag=flag+1
             break

        cv2.putText(img, str(id) + " " + str(conf), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (120, 255, 120), 1)

        cv2.imshow('frame', img)
    #cv2.imshow('gray',gray);
    if flag == 10:
        playsound('transactionSound.mp3')
        print("Transaction Blocked")
        break;
    if time.time()>start+period:
        break;
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cap.release();
cv2.destroyAllWindows();
