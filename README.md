# opencv를 통한 임베디드 제어와 사용자 인증 ,구글클라우드 연동
라즈베리파이를 이용한 임베디드 api활용 upload to googledrive


1)각 사람별로 구별_데이터 수집
import cv2
import os


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
   
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()






2)개인의 얼굴 학습 코드.
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml");

# function to get the images and label data

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml

recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the number of faces trained and end program

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))






3)각 사람별로 구별하기_인식
# -*- coding: utf-8 -*-
import cv2
import time
import datetime
import numpy as np
import os
import RPi.GPIO as GPIO



GPIO.setwarnings (False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(14, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)


recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')

cascadePath = "haarcascade/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX



#iniciate id counter

id = 0



# names related to ids: example ==> loze: id=1,  etc


names = ['TY', 'loze', 'ljy', 'chs', 'ksw']



# Initialize and start realtime video capture

cam = cv2.VideoCapture(0)

cam.set(3, 640) # set video widht

cam.set(4, 480) # set video height
count = 0


# Define min window size to be recognized as a face

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # unFlip vertically

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
        now = datetime.datetime.now().strftime("%d_%H-%M-%S")

        if (confidence < 100):

            id = names[id]

            confidence = "  {0}%".format(round(100 - confidence))
            GPIO.output(14, True)
            time.sleep(1)
            GPIO.output(14, False)
            time.sleep(4)
            GPIO.output(14, True)
            time.sleep(4)
            GPIO.output(14, False)
            time.sleep(1)
            GPIO.output(17, False)

            cam.release()
        else:

            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            GPIO.output(17,True)
             time.sleep(10)
            GPIO.output(17,False)
            time.sleep(2)
            GPIO.output(14, False)
            time.sleep(0)
            GPIO.output(14, False)
            time.sleep(0)

            cv2.imwrite("/home/pi/J/"+str(now)+".png",img)
            cv2.imshow('image', img)
            cam.release()



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
GPIO.cleanup()
                              
                              
                              
4)Linux 명령어 rclone 을 통해 구글드라이브와 sync 하기.
처음  아래 명령여를 순차적으로 실행한다.
$ curl -L https://raw.github.com/pageauc/rclone4pi/master/rclone-install.sh | bash
bash ver 1.6 written by Claude Pageau
-------------------------------------------------------------------------------
--2019-04-29 16:00:53--  https://downloads.rclone.org/rclone-current-linux-arm.zip
...
...
...
...
rclone installed at /usr/bin/rclone
-------------------------------------------------------------------------------
                 INSTRUCTIONS Google Drive Example

1 You will be required to have a login account on the remote storage service
  Open putty SSH login session to RPI and execute command below

  rclone config

  Follow rclone prompts. For more Details See
  https://github.com/pageauc/rclone4pi/wiki/Home
2 At name> prompt specify a reference name  eg gdmedia
3 At storage> prompt Enter a remote storage number from List
4 Select Auto Config, At Link: prompt, left click
  and highlight rclone url link (do not hit enter)
5 on computer web browser url bar right click paste and go.
6 On computer web browser security page, Confirm access.
7 Copy web browser access security token and paste
  into RPI SSH session rclone prompt. Enter to accept
8 To test remote service access. Execute the following where
  gdmedia is the name you gave your remote service

  rclone ls gdmedia:/

Example sync command make source identical to destination

rclone sync -v /home/pi/rpi-sync gdmedia:/rpi-sync

To upgrade

  cd rpi-sync
  ./rclone-install.sh upgrade

For more Details See https://github.com/pageauc/rclone4pi/wiki/Home
Bye 
-위와 같은 화면이 설치되었다고 결과가 나온다.
$ rclone config
2019/04/29 16:21:29 NOTICE: Config file "/home/pi/.config/rclone/rclone.conf" not found - using defaults
No remotes found - make a new one
n) New remote
s) Set configuration password
q) Quit config
n/s/q> n 
순차적으로 진행시켜준다.
name> drive(이름은 예시로 들었다 아무거나 설정 가능)
Type of storage to configure.
Enter a string value. Press Enter for the default ("").
Choose a number from below, or type in your own value
 1 / A stackable unification remote, which can appear to merge the contents of several remotes
   \ "union"
 2 / Alias for a existing remote
   \ "alias"
 3 / Amazon Drive
   \ "amazon cloud drive"
 4 / Amazon S3 Compliant Storage Provider (AWS, Alibaba, Ceph, Digital Ocean, Dreamhost, IBM COS, Minio, etc)
   \ "s3"
 5 / Backblaze B2
   \ "b2"
 6 / Box
   \ "box"
 7 / Cache a remote
   \ "cache"
 8 / Dropbox
   \ "dropbox"
 9 / Encrypt/Decrypt a remote
   \ "crypt"
10 / FTP Connection
   \ "ftp"
11 / Google Cloud Storage (this is not Google Drive)
   \ "google cloud storage"
12 / Google Drive
   \ "drive"
13 / Hubic
   \ "hubic"
14 / JottaCloud
   \ "jottacloud"
15 / Koofr
   \ "koofr"
16 / Local Disk
   \ "local"
17 / Mega
   \ "mega"
18 / Microsoft Azure Blob Storage
   \ "azureblob"
19 / Microsoft OneDrive
   \ "onedrive"
20 / OpenDrive
   \ "opendrive"
21 / Openstack Swift (Rackspace Cloud Files, Memset Memstore, OVH)
   \ "swift"
22 / Pcloud
   \ "pcloud"
23 / QingCloud Object Storage
   \ "qingstor"
24 / SSH/SFTP Connection
   \ "sftp"
25 / Webdav
   \ "webdav"
26 / Yandex Disk
   \ "yandex"
27 / http Connection
   \ "http"
Storage> 12 이때 사용할것은 구글드라이브이므로 12번을 선택해준다.

Google Application Client Id
Setting your own is recommended.
See https://rclone.org/drive/#making-your-own-client-id for how to create your own.
If you leave this blank, it will use an internal key which is low performance.
Enter a string value. Press Enter for the default ("").
client_id>
Google Application Client Secret 
Setting your own is recommended.
Enter a string value. Press Enter for the default ("").
client_secret>// 아이디를 설정해서 속도를 더 빠르게 할 수 있지만 여기서 설정하진 않는다.
Scope that rclone should use when requesting access from drive.
Enter a string value. Press Enter for the default ("").
Choose a number from below, or type in your own value
 1 / Full access all files, excluding Application Data Folder.
   \ "drive"
 2 / Read-only access to file metadata and file contents.
   \ "drive.readonly"
   / Access to files created by rclone only.
 3 | These are visible in the drive website.
   | File authorization is revoked when the user deauthorizes the app.
   \ "drive.file"
   / Allows read and write access to the Application Data folder.
 4 | This is not visible in the drive website.
   \ "drive.appfolder"
   / Allows read-only access to file metadata but
 5 | does not allow any access to read or download file content.
   \ "drive.metadata.readonly"
scope> 1 //rclone 의 권한을 설정한다.
이후에 아래와 같은 선택창을 표시한다.
1.Application Data folder를 제외한 모든 파일에 대해 access 가능
2.파일 내용을 읽기만 가능
3.Rclone에서만 만들어진 파일만 access 가능
4.Application Data folder access 가능
5.오직 파일의 metadata만 접근 가능(다운로드나 읽기 불가) //여기선 1번을 설정했다.

ID of the root folder
Leave blank normally.
Fill in to access "Computers" folders. (see docs).
Enter a string value. Press Enter for the default ("").
root_folder_id> 1Sb6********************* <- 여기서 나오는 id가 구글 드라이브 폴더뒤에 있는 id를 받아와 적어주면 된다.
Service Account Credentials JSON file path
Leave blank normally.
Needed only if you want use SA instead of interactive login.
Enter a string value. Press Enter for the default ("").
service_account_file>
Edit advanced config? (y/n)
y/n> n
Use auto config?
y/n> n //전체적으로 설정에 관해 잘 모를땐 가장 기본으로 설정해준다.
If your browser doesn't open automatically go to the following link: https://accounts.google.com/o/oauth2/auth?/*****************************
Enter verification code> 4/************************** <- 이후 링크가 나오고 타고 들어가면, 아이디를 적게끔 창이 나온다 이것을 복사하여 붙여주자.
Configure this as a team drive?
y) Yes
n) No
y/n> y
Fetching team drive list...
Choose a number from below, or type in your own value
 1 / "팀드라이브 이름"
   \ "0A****************"
Enter a Team Drive ID> 1
--------------------
# 설정한 rclone 정보
--------------------
y) Yes this is OK
e) Edit this remote
d) Delete this remote
y/e/d> y
Current remotes:

Name                 Type
====                 ====
"드라이브 이름"       drive  // 이와같이 화면이 나오면 설정한 것을 확인 할 수 있다
$ rclone config
Current remotes:

Name                 Type
====                 ====
gdrive               drive

e) Edit existing remote
n) New remote
d) Delete remote
r) Rename remote
c) Copy remote
s) Set configuration password
q) Quit config
e/n/d/r/c/s/q> e
Choose a number from below, or type in an existing value
 1 > gdrive
remote> 1 //설치와 설정이 완료된 모습을 보이고 있다.
$ rclone copy 라즈베리파이 내부경로의 디렉터리 경로를 적는다.:Raspberry Pi내부의 구글 드라이브를 저장하는 공간 (여기선 drive) : 구글 드라이브 폴더명을 적어준다.
이렇게 함으로서 Raspberry Pi안에 있는 데이터가 지정한 구글드라이브 계정안으로 이미지가 백업됨을 알 수 있다.

이와같은 결과가 나옴을 알 수 있다.
5)Crontab 명령을 이용한 rclone 자동실행.
$ crontab -e // 확인함으로 편집공간으로 넘어간다.
이후 사진과 같은 영역이 보이고 아래에 * * *＊＊ 표시를 한 후 {rclone copy 라즈베리파이 내부경로의 디렉터리 경로를 적는다.:Raspberry Pi내부의 구글 드라이브를 저장하는 공간 (여기선 drive) : 구글 드라이브 폴더명을 적어준다.} 내용을 적어주고 저장하면
자동적으로 corntab이 내부에서 자동으로 명령어를 실행하여 매분 구글 드라이브로 업로드 하는것을 알 수 있다.
이와 같은 결과가 나옴을 알 수 있으며,
아래에서 파일명을 현재 시간으로 지정을 하여 업로드하면 구글드라이브에서도 같으 파일명을 공유받은것을 알 수 있다.



<참고 자료 & 문헌>
-Real-Time Face Recognition: An End-to-End Project/Hackster.io // OpenCv 학습
-https://jdm.kr/blog/2 // crontab 설정.
-https://lukael.kr// Google Drive sync&copy&move 설정.
-https://www.raspberrypi.org/forums/viewtopic.php?t=141869 //GPIO 사용 참고
-https://github.com/dltpdn/opencv-for-rpi// OpenCv 다운 및 설정.
-[1] https://docs.opencv.org/4.1.0/d7/d8b/tutorial_py_face_detection.html
 [2] https://eehoeskrap.tistory.com/95 
 [3] http://www.willberger.org/cascade-haar-explained/ // Harr Cascades 사용 참고.
- https://github.com/opencv/opencv/tree/master/data/haarcascades //Harr Cascades 데이터 다운                             
                              
