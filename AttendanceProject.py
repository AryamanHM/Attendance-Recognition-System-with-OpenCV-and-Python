import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendance'; #import images
images=[]  #create list of all images we will import
classNames=[]   #write names of images
myList=os.listdir(path)   #grab lists of images in folder
print(myList)       #print the names of images in directory
for cl in myList:     #use names and import images one by one
    curImg=cv2.imread(f'{path}/{cl}')     #read current image which is basically path.cl is name of image first will be Bill Gates.jpg
    images.append(curImg) #append current image
    classNames.append(os.path.splitext(cl)[0]) #append class name and inside here.we dont want to append BillGates.jpg we just want Bill Gates.Grab first element to print Bill Gates
print(classNames) #printing image names without extension of files

def findEncodings(images):  #to compute all encodings for us.It wud require list of images
    encodeList=[] #empty list to contain all encodings in end
    for img in images: #loop through images
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert to RGB
        encode = face_recognition.face_encodings(img)[0]  #find encodings
        encodeList.append(encode) #append it to list
    return encodeList   #return the list

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines();
        nameList=[]
        #print(myDataList) gives ['Name,Time']
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])  #will be name
            if name not in nameList:
                now=datetime.now();
                dtString=now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


#markAttendance('Elon') bought Elon with current time in attendance.csv


encodeListKnown=findEncodings(images)  #call function for known faces
#print(len(encodeListKnown)) #would print 3 for number of testing images
print('Encoding Complete')  #done step
cap = cv2.VideoCapture(0) #find matches between our encodings we dont have image to match it woth,image would be coming from webcam
#0 is ID

while True:
    success,img=cap.read() #give us our image
    imgS=cv2.resize(img,(0,0),None,0.25,0.25) #1/4th of size
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) #convert to RGB

    facesCurFrame=face_recognition.face_locations(imgS) #findall locations in small image
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)#finding current frame-send small image and send
    #faces

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame): #iterate through all faces that we found in current frame and comapre faces to encodings we found before
    #one by one it will grab 1 face location from faces current frame list and it will grab encodings
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


