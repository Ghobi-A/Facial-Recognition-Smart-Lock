#!/usr/bin/env python
'''
[Original author header redacted]
Raspberry pi facial detection system and GUI for additional face detection
2022
'''
from tkinter import *
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter import messagebox
from PIL import Image,ImageTk
import subprocess
import os
from threading import Thread
from config import CONFIG
from config import set_config
import shutil
import logging
import sys
import cv2
from imutils import paths
import imutils
import face_recognition
import pickle
import RPi.GPIO as GPIO
import time

class SpecButton(Button):
    '''class used to enhance standard Button class to change button colors when hovering with the mouse'''
    def __init__(self, *args, **kwargs):
        Button.__init__(self, *args, **kwargs)
        self.bind("<Enter>", self.changeBGEnter)

    def changeBGEnter(self, event):
        self.config(activebackground='red')
        self.config(activeforeground='white')

  


class View(Frame):
        ##################### INIT #######################################
    def __init__(self, contr):
        #setup program mode
        if os.path.exists('encodings.pickle'):
            self.mode = 'faceRecognition'  #default startup mode
        else:
            self.mode = 'takePictures'  #default startup mode without faces
        self.prevMode = 'init'
        
        #setup logger
        self.logger = logging.getLogger(__name__)

        self.programDir = os.getcwd()

        self.contr = contr #link to controller class
        self.top=Toplevel()
        self.foreground = CONFIG.get('GUI', 'foregroundcolor')
        self.background = CONFIG.get('GUI', 'backgroundcolor')

        #display size = 800*480, need 35 pixels space for program
        self.top.geometry('800x480')

        style = ttk.Style()
        style.layout('TNotebook.Tab', []) # turn off tabs
        self.note = ttk.Notebook(self.top)

        self.tab1 = Frame(self.note)

        self.note.add(self.tab1, text = "Main")

        self.note.columnconfigure(0, weight=1)
        self.note.pack(expand=1, fill=BOTH)

        self.frame1 = Frame(self.tab1)
        self.frame1.place(x = 0, y = 0, width=800, height=480)
        
        #load the widgets
        self.main_view()

        #setup GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(23, GPIO.OUT)

        #open camera object
        self.cam=cv2.VideoCapture(0)
        time.sleep(2)
        print('Camera Initialised')
        # Start the thread to read frames from the video stream
        self.thread1 = Thread(target=self.readCam, args=())
        self.thread1.daemon = True
        self.thread1.start()
        time.sleep(2)
        # Start doorlock thread
        self.lockOpen = False        
        self.thread2 = Thread(target=self.doorOpen, args=())
        self.thread2.daemon = True
        self.thread2.start()

    def main_view(self):
        ##################### MAIN VIEW #######################################
        #define frames
        self.frame1Left = Frame(self.frame1)
        self.frame1Left.place(x = 0, y = 0, width=100, height=480)
        self.frame1Mid = Frame(self.frame1)
        self.frame1Mid.place(x = 100, y = 0, width=500, height=480)
        self.frame1Right = Frame(self.frame1)
        self.frame1Right.place(x = 600, y = 0, width=200, height=480)


        #buttons
        noButtons = 2
        self.button = ['']* noButtons
        buttonText = ['Add\na face','']  #'Del\nFaces'  not used yet
        buttonCommand = [self.button01Action, self.button02Action]
        for i in range (noButtons):
            self.button[i] = SpecButton(self.frame1Left, text=buttonText[i], command=buttonCommand[i])
            self.button[i].config(font=("Courier bold", 12))
            self.button[i].pack(side = TOP, fill= BOTH, expand = 1)

        #add canvas to vidoFrame
        self.videoCanvas = Canvas(self.frame1Mid,width=700,height=480, bg = 'blue')
        self.videoCanvas.place(x=0, y=0)
    
        #information to user
        self.info = Text(self.frame1Right, fg=self.foreground, bg=self.background,wrap='word')
        self.info.config(font=("Courier", 14))
        self.info.pack(side = LEFT, fill= BOTH, expand = 1)
        self.info.insert(END,'')

    ##################### routines #######################################
    def update(self):
        '''
        main update routine, 
        triggered every 100 msec from main program
        '''

        if self.mode == 'faceRecognition':
            if self.mode != self.prevMode:
                #Initialize 'currentname' to trigger only when a new person is identified.
                self.currentname = "unknown"
                #Determine faces from encodings.pickle file model created from train_model.py
                encodingsP = "encodings.pickle"

                # load the known faces and embeddings along with OpenCV's Haar
                # cascade for face detection
                print("[INFO] Loading encodings + Face Detector...")
                self.data = pickle.loads(open(encodingsP, "rb").read())
                self.prevMode = self.mode

            self.faceRecognition()

        elif self.mode == 'takePictures':
            if self.mode != self.prevMode:
                self.info.delete('1.0', END)
                self.infoMessage('Machine Learning\nNew Face\n\nTake Minimum 10\npictures')
                self.prevMode = self.mode

            self.takePictures()
        else:
            pass

    def button01Action(self):
        if self.button[0]['text'] == 'Add\nFaces':
            self.button[0]['text'] =  'Take\nPicture'
            self.button[1]['text'] =  'Start'
   
            #create new dataset to store new face
            datasetDir = self.programDir + '/dataset'
            print(datasetDir)

            if not os.path.exists(datasetDir):
                os.makedirs(datasetDir)

            #directory nummer filled with leading zeros
            self.firstDirNum = 1
            dirNum = str(self.firstDirNum)
            dirNum = dirNum.zfill(3)
            
            self.firstDir = 'face' + dirNum
            dirList = os.listdir(datasetDir)

            #check if directory exists
            while self.firstDir in dirList:
                print('existing dir', self.firstDir)
                self.firstDirNum+=1
                dirNum = str(self.firstDirNum)
                dirNum = dirNum.zfill(3)
                self.firstDir = 'face' + dirNum
            
            print('storage dir:', self.firstDir)
            #Create storage directory
            faceDir = datasetDir + '/' + self.firstDir
            os.mkdir(faceDir)

            #take picture
            print('Show the frames to take picture from')
            self.takePicturesReady = False
            self.name = self.firstDir #replace with real name if keyboard available

            self.img_counter = 0
            #self.interval = 100 #new frame every 100 msec
            self.mode = 'takePictures'

        else:  #save picture
                img_name = "dataset/"+ self.name +"/image_{}.jpg".format(self.img_counter)
                cv2.imwrite(img_name, self.image)
                self.infoMessage('Picture {} written!'.format(img_name))
                print("{} written!".format(img_name))
                self.img_counter += 1

    def button02Action(self):
        if self.button[1]['text'] == '':  #'Del\nFaces'  not in use
            print('del faces')  # not in use yet
        else: #all pictures taken
            print('All pictures taken')
            self.takePicturesReady = True
            self.trainModel()
            self.button[0]['text'] = 'Add\nFaces'
            self.button[1]['text'] = ''  #'Del\nFaces'  not used at the moment
            self.mode = 'faceRecognition'

    #will run every 100 msec
    def faceRecognition(self):
        # loop over frames from the video file stream
          # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        if self.status:
            frame = self.frame
 
        frame = imutils.resize(frame, width=500)
        # Detect the fce boxes
        boxes = face_recognition.face_locations(frame)
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []
        self.info.delete('1.0', END)
        self.infoMessage('Face\nRecognition')
        
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.data["encodings"],
                encoding)
            name = "Unknown" #if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

                #If someone in your dataset is identified, print their name on the screen
                if self.currentname != name:
                    self.currentname = name
                    print(self.currentname)

            # update the list of names
            names.append(name)
        # if a face is recognized open the door                
        if names:
            #open door
            self.lockOpen = True

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (0, 255, 255), 2)

        # display the image to our screen
        self.pillowImage = ImageTk.PhotoImage(Image.fromarray(frame)) # to ImageTk format
        # Update image
        self.videoCanvas.create_image(0, 0, anchor=NW, image=self.pillowImage)


    def takePictures(self):
        '''take pictures from new faces'''
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        if self.status:
            frame = self.frame
        self.image = imutils.resize(frame, width=500)
        #convert to Tkinter pillow
        self.pillowImage = ImageTk.PhotoImage(Image.fromarray(self.image)) # to ImageTk format
        # Update image
        self.videoCanvas.create_image(0, 0, anchor=NW, image=self.pillowImage)
        # Update image
    
        
    def trainModel(self):
        '''
        train model with new face pictures
        '''
        # our images are located in the dataset folder
        print("[INFO] start processing faces...")
        imagePaths = list(paths.list_images("dataset"))

        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []
        self.info.delete('1.0', END)
        self.infoMessage('TRAIN MODEL\n\n')
        self.infoMessage("[INFO] start processing faces...")
        self.top.update()
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            self.infoMessage("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
            self.top.update()
            print("[INFO] processing image {}/{}".format(i + 1,
                len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the input image and convert it from RGB (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb,
                model="hog")

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                knownEncodings.append(encoding)
                knownNames.append(name)

        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()

    def doorOpen(self):
        while True:
            if self.lockOpen:
                print('Door lock initalised')
                self.infoMessage('\nFace is known in the database\nAuthorised access!!')
                GPIO.output(23, GPIO.HIGH)
                time.sleep(5)
                self.lockOpen = False
            else:
                GPIO.output(23, GPIO.LOW)
            time.sleep(.1) 

    def readCam(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.cam.isOpened():
                (self.status, self.frame) = self.cam.read()
    
    def infoMessage(self, message):
        self.info.insert(END,message + '\n')
        self.info.see(END)