import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'photos'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        f.seek(0, os.SEEK_END)  # Move the file pointer to the end of the file

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

'''
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
'''                

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)



'''

import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime


video_capture=cv2.VideoCapture(0)   # for capturing video

# Batman_image = face_recognition.load_image_file("photos/Batman.jpg")
# Batman_encoding = face_recognition.face_encodings(Batman_image)[0]
 
jobs_image = face_recognition.load_image_file("photos/Steve.jpg")       # loaded image from file
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]          

tata_image = face_recognition.load_image_file("photos/Tata.jpg")
tata_encoding = face_recognition.face_encodings(tata_image)[0]



# image_path= "photos/Batman.jpg"

# image = cv2.imread(image_path)

# Check if the image was loaded successfully
#if image is not None:
#    cv2.imshow("Loaded Image", image)  # Display the image in a window
#    cv2.waitKey(0)  # Wait for a key press
#    cv2.destroyAllWindows()  # Close the image window
#else:
#    print("Image not loaded or invalid path.")



known_faces_encoding=[
    jobs_encoding,
    tata_encoding,
#   Batman_encoding,
]

known_faces_name=[
    "jobs",
    "tata",
#  "Batman",
]

students=known_faces_name.copy()

face_locations=[]
face_encoding=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")


f=open(current_date+'.csv','w+',newline='')  # open method and write in csv file f. w+ method write method
lnwriter=csv.writer(f)                       # write in f file

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)     # resize frame
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
      face_locations=face_recognition.face_locations(rgb_small_frame)
      face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations) 

      face_names = []

      for face_encoding in face_encodings:
           matches=face_recognition.compare_faces(known_faces_encoding,face_encoding)
           name=""

           face_distance=face_recognition.face_distance(known_faces_encoding,face_encoding)
          
           best_match_index=np.argmin(face_distance)        # get the best matching solution
           if matches[best_match_index]:
                name=known_faces_name[best_match_index]

           face_names.append(name)      
           
           if name in known_faces_name:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("attendance system", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()    
      

'''