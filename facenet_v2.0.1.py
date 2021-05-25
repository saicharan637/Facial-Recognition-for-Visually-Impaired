from multiprocessing.connection import Client
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
import sys
import win32com.client as wincl
import pickle
import csv

PADDING = 50
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")
g_start_time = start_time  = time.time()
SVM = False

def img_path_to_encod(image_path):
    img1 = cv2.imread(image_path, 1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    part_image = None
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING
        height, width, channel = img1.shape
        part_image = img1[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if part_image is None:
        print(image_path,'image is not a face')
    return img_to_encod(part_image)
    
def img_to_encod(image):
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    address = ('localhost', 6000)
    conn = Client(address, authkey=str.encode('secret password'))
    conn.send('now')
    conn.send(img)
    while True:
        msg = conn.recv()
        if(msg == 'encc'):
            embedding = conn.recv()
        else:
            if msg == 'close':
                conn.send('close')
                conn.close()
                break
    conn.close()
    return embedding

def prepare_database():
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encod(file)
    pickle.dump( database, open( "trainedData/emds.p", "wb" ) )
    print(database)
    return database
    
def prepare_csv_n_database():
    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encod(file)
    with open('trainedData/f1.csv', 'w') as f:
        w = csv.writer(f)
        for key, value in database.items():
            temp = value.tolist()[0]
            temp.append(key)
            w.writerow(temp)
    return database
    
def load_database():
    database = pickle.load( open( "trainedData/emds.p", "rb" ) )
    return database
    
def webcam_face_recognizer(database):
    global ready_to_detect_identity
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def process_frame(img, frame, face_cascade):
    global ready_to_detect_identity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING
        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        identity = find_identity(frame, x1, y1, x2, y2)
        if identity is not None:
            identities.append(identity)
    if identities != []:
        print('called audio' )
        ready_to_detect_identity = False
        pool = Pool(processes=1) 
        # We run this as a separate process so that the camera feedback does not freeze
        pool.apply_async(welcome_users, [identities])
        #welcome_users(identities)
    return img

def find_identity(frame, x1, y1, x2, y2):
    print('in find identity')
    height, width, channel = frame.shape
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    return who_is_it(part_image, database)

def who_is_it(image, database):
    if(SVM):
        return who_is_it_SVM(image)
    else:
        print('in who is it')
        start_time  = time.time()
        encoding = img_to_encod(image)
        elapsed_time = time.time() - start_time    
        min_dist = 100
        identity = None
        for (name, db_enc) in database.items():
            dist = np.linalg.norm(db_enc - encoding)
            print('distance for %s is %s' %(name, dist))
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > 0.52:
            return None
        else:
            return str(identity)

def who_is_it_SVM(image):
    print('in who is it')
    start_time  = time.time()
    encoding = img_to_encod(image)
    elapsed_time = time.time() - start_time

    return sss

def welcome_users(identities):
    global ready_to_detect_identity
    welcome_message = 'Hello, '
    if len(identities) == 1:
        welcome_message += '%s.' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += '.'
    windows10_voice_interface.Speak(welcome_message)
    ready_to_detect_identity = True

def test_recogniser(database):
    img = cv2.imread('S:\\photods\\S  0747a 50.jpg',1)
    frame = img
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = process_frame(img, frame, face_cascade)   
    cv2.imshow("preview", img)
    key = cv2.waitKey(0)
    cv2.destroyWindow("preview")

def train_svm():
    return
    
def clsshndlr(arg):
    if(arg[0]) == '--svm':
        SVM = True
        train_svm()

if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) > 0:
        if arg[0] == '-t':
            database = prepare_database()
            print('pickle created in ')
            elapsed_time = time.time() - g_start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        elif arg[0] == '-tc':
            database = prepare_csv_n_database()
            print('csv created in ')
            elapsed_time = time.time() - g_start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        elif arg[0] == '-tm':
            clsshndlr(arg[1:])
        elif arg[0] == '-rc':
            database = load_database_csv()
            load_model()
            print('loaded the csv in')
            elapsed_time = time.time() - g_start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            # webcam_face_recognizer(database)
            test_recogniser(database)
        else:
            database = load_database()
            print('loaded the database in')
            elapsed_time = time.time() - g_start_time
            print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            # webcam_face_recognizer(database)
            test_recogniser(database)
    else:
        print('Usage : python facenet_v2.0.1.py -t/-r/-tc/-rc')

