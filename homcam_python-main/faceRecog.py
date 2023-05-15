import cv2
import numpy as np
import os
from time import sleep
import datetime
import sys, os
import requests
from PIL import Image
import pyrebase
from firebase import firebase

import firebase_admin

from firebase_admin import credentials, firestore, storage

from firebase_admin import db


from PIL import Image

from uuid import uuid4

import schedule



import time 
import schedule

from pyfcm import FCMNotification
from glob import glob
import urllib.request as req



PROJECT_ID = "android-2305a"

cred = credentials.Certificate("firebase JSON") 

firebase_admin.initialize_app(cred,{

    'databaseURL':"Firebase 주소",

    'storageBucket':f"{PROJECT_ID}.appspot.com"

    })



config={

    "apiKey":"WEBKEY", # webkey

    "authDomain":"projectID", # projectID

    "databaseURL":"DB URL", 

    "storageBucket":"storageURL", # storageURL

    "serviceAccount":"json"

    }



fb=pyrebase.initialize_app(config)



database=fb.database()

storage2=fb.storage()

bucket = storage.bucket()

ref=firestore.client()

#bucket=storage.bucket("/video")



#uploadfile="/makevideo"



#storage=firebase.storage()



#popupMessage send

def sendMessage(body, title):

    ref = db.reference('Token/value')

    #print(ref.get())

    APIKEY = "API KEY"

    TOKEN = str(ref.get())

# 파이어베이스 콘솔에서 얻어 온 서버 키를 넣어 줌

    push_service = FCMNotification(APIKEY)

    # 메시지 (data 타입)

    data_message = {

        "body": body,

        "title": title

    }

    # 토큰값을 이용해 1명에게 푸시알림을 전송함

    result = push_service.single_device_data_message(registration_id=TOKEN, data_message=data_message)

    # 전송 결과 출력

    print(result)

    

  

def fileUpload(file):

    blob = bucket.blob('video/'+file)

    new_token = uuid4()

    metadata = {"firebaseStorageDownloadTokens": new_token} 

    blob.metadata = metadata

    #upload file

    blob.upload_from_filename(filename='./'+file, content_type='video/avi') 

    #debugging hello

    print("hello ")

    print(blob.public_url)

 

def execute_camera(videoname):

    fileUpload(videoname)





def Imagedownload():

    spath="./data/"

    [os.remove(f) for f in glob("/home/pi/1109/dataset/*")]



    ref = db.reference()

    a=list(ref.child("users/").get())

    names1 = ['None']

    names = names1 + a



    for k in range(1,len(names)):

        piclis=list(ref.child("users/"+names[k]+"/photolink").get())

        spath="./dataset/"

        print(piclis)

        j=1

        for i in piclis:

            puturl=str(ref.child("users/"+names[k]+"/photolink/"+i).get())

            print(puturl)

            download_file=req.urlretrieve(puturl,spath+"user."+str(k)+"."+str(j)+".jpg")

            j+=1





def Imagecrop():

    cam = cv2.VideoCapture(0)

    cam.set(3, 640) # set video width

    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    path = 'dataset'

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     

    for imagePath in imagePaths:

        img = cv2.imread(imagePath)

        #img = cv2.flip(img, -1) # flip video image vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     

            count += 1

            # Save the captured image into the datasets folder

            id = int(os.path.split(imagePath)[-1].split(".")[1]) 

            cv2.imwrite("data/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video

        if k == 27:

            break

        elif count >= 100: # Take 100 face sample and stop video

            break

    # Do a bit of cleanup

    print("\n [INFO] Exiting Program and cleanup stuff")
    cv2.destroyAllWindows()


def ImageTraining():
# Path for face image database
    path = 'dataset'
    detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml");
    recognizer = cv2.face.LBPHFaceRecognizer_create()

# function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1]) # value is int
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

def LoadImage():
    Imagedownload()
    Imagecrop()
    ImageTraining()
    ref = db.reference()
    a=list(ref.child("users/").get())
    names1 = ['None']
    names = names1 + a 
    

# Save the model into trainer/trainer.yml

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX



#iniciate id counter



# names related to ids: example ==> loze: id=1,  etc

# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.



ref = db.reference()
a=list(ref.child("users/").get())
names = ['None','Suzy']
#names = names1 + a



# Initialize and start realtime video capture

cam = cv2.VideoCapture("http://192.168.137.13:8080/?action=stream")

cam.set(3, 640) # set video widht

cam.set(4, 480) # set video height



# Define min window size to be recognized as a face

minW = 0.1*cam.get(3)

minH = 0.1*cam.get(4)

#LoadImage()
id = 0
unknown_count=0
known_count=0

while True:

    schedule.every(30).minutes.do(LoadImage)
    #schedule.every(10).seconds.do(LoadImage)
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 

        gray,

        scaleFactor = 1.2,

        minNeighbors = 5,

        minSize = (int(minW), int(minH)),

       )



    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        
        # Check if confidence is less them 100 ==> "0" is perfect match
        
        if (confidence < 103):
            known_count += 1
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            print(id)
            if(known_count>=20):
                getSelect=ref.child("users/"+id).child("select").get()
                print(getSelect)
                if(getSelect=="등록"):
                    sendMessage("message",id+"님이 감지되었습니다!" )
                    known_count=0
                
        else:

            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            unknown_count=unknown_count+1;                       
            print("move")
            if(unknown_count>=20):
                now = datetime.datetime.now()
                fname=now.strftime('%Y-%m-%d %H:%M:%S')
                vname='makevideo/'+fname+'.avi'

            #cap = cv2.VideoCapture(-1)
                fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
                out = cv2.VideoWriter(vname, fcc, 20.0, (640,480))#fcc-1

                max_time_end = time.time()+(60*1/2)#60 is 15sec

                ref = db.reference()

                ref.child("video/makevideo/").child(fname).child("filename").set(fname+'.avi')

                #ref.child("Noti").child("data").set("1")

                sendMessage("message","낯선 사람이 침입했습니다!!!!" )



                while(cam.isOpened()):

                    ret2, frame2 = cam.read()#cap

                    if ret2==True:

                        frame2 = cv2.flip(frame2,-1)
                        out.write(frame2)


                        if time.time() > max_time_end:

                            print("video saved")

                            execute_camera(vname)#record video and upload DB
                            unknown_count=0
                            break

                    else:break
                out.release()

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video

    if k == 27:

        break

# Do a bit of cleanup

print("\n [INFO] Exiting Program and cleanup stuff")

cam.release()

cv2.destroyAllWindows()




