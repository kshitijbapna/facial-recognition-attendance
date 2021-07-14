import os
import cv2
import pandas as pd
import face_recognition as fr

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if ret==False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Registration', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        S_name = input("Enter the student name:")
        filename = S_name+".jpg"
        path = './Images/'
        cv2.imwrite(os.path.join(path , filename), frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

Attendance = pd.DataFrame()
Attendance["Name"]=0
path = './Images/'
ImagesList = os.listdir(path)
for filename in ImagesList:
    name = filename[:-4]
    data = [{'Name':f'{name}'}]
    Attendance.loc[len(Attendance.index)]=list(data[0].values())
    
known_face_encodings = []
known_face_names = []

path = './Images/'
imagesList = os.listdir(path)
for filename in imagesList:
    image = fr.load_image_file(f"./Images/{filename}")
    name = filename[:-4]
    known_face_names.append(name)
    known_face_encodings.append(fr.face_encodings(image)[0])
    
import pickle

with open("known_face_names.txt", "wb") as fp:   #Pickling
   pickle.dump(known_face_names, fp)
with open("known_face_encodings.txt","wb") as fp:
    pickle.dump(known_face_encodings,fp)

Attendance["Total"]=0
Attendance["Percentage"]=0
Attendance.to_csv("Attendance List.csv",index=False)