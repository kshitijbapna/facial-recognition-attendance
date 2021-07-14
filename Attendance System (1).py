import cv2
import time
import face_recognition as fr
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from datetime import datetime
import os
import pandas as pd
import pickle

TIME = float(input("Enter class duration(in minutes):"))
Pre_Time = float(input("Enter the required minutes for attendance:"))
Count=0
Total=0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

dt = datetime.now()
d1 = dt.strftime("%d-%m-%Y %H;%M")
print("Filename:",d1)
directory = f"{d1}"
path = os.path.join(".", directory) 
os.mkdir(path) 

known_face_encodings = []
known_face_names = []
Attendance = pd.read_csv("./Attendance List.csv")
Attendance[f"{d1}"] = 0

path = './Images/'
# imagesList = os.listdir(path)
# for filename in imagesList:
#     image = fr.load_image_file(f"./Images/{filename}")
#     name = filename[:-4]
#     known_face_names.append(name)
#     known_face_encodings.append(fr.face_encodings(image)[0])

with open("known_face_encodings.txt", "rb") as fp:   # Unpickling
  known_face_encodings = pickle.load(fp)
with open("known_face_names.txt", "rb") as fp:   # Unpickling
  known_face_names = pickle.load(fp) 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video_capture = cv2.VideoCapture(0)
t_end = time.time()
face_locations = []
face_encodings = []
face_names = []
t=0
process_this_frame = True

start = time.time()
while(True):
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if ret==False:
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face in face_encodings:
            match = fr.compare_faces(known_face_encodings, face,tolerance=0.5)
            name = "Unknown"
            face_distances = fr.face_distance(known_face_encodings,face)
            best_match_index = np.argmin(face_distances)
            if match[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    c = 0
    for (top,right,bottom,left),name in zip(face_locations,face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        c = 0
        for i in range(0,len(Attendance)):
            if Attendance["Name"][i]==name:
                t+=0.5
                print(t)
                if t/130>=Pre_Time and Total>=(10*Pre_Time):
                    Attendance[f"{d1}"][i]=1
                filename = name+".jpg"
                path = f'./{d1}'
                cv2.imwrite(os.path.join(path , filename), frame)        
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye) 
        ear = (leftEAR + rightEAR) / 2.0

        if ear<0.23:
            Count+=1
        else:
            if Count >= 1:
                Total+=1
            Count=0
        
        cv2.putText(frame, "Blinks: {}".format(Total), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Attendance System", frame)
    if time.time()>=start+(TIME*60):
        for i in range(0,len(Attendance)):
            Attendance["Total"][i]+=Attendance[f"{d1}"][i]
            Attendance["Percentage"][i]=(Attendance["Total"][i]/(len(Attendance.columns)-3))*100 
        break
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    #time.sleep(0.02)

#for i in range(0,len(Attendance)):
#    Attendance["Total"][i]+=Attendance[f"{d1}"][i]
#    Attendance["Percentage"][i]=(Attendance["Total"][i]/(len(Attendance.columns)-3))*100 
    
Attendance.to_csv("Attendance List.csv", index=False)    
video_capture.release()
cv2.destroyAllWindows() 