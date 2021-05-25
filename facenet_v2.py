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
from keras.models import load_model
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl

PADDING = 50
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")
g_start_time = start_time  = time.time()
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

def handler(argg):
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    print('Compilation begins now')
    start_time = time.time()
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    elapsed_time = time.time() - start_time
    start_time=time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print('loading weights now..')
    load_weights_from_FaceNet(FRmodel)
    elapsed_time = time.time() - start_time
    start_time=time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

def img_path_to_encod(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)
    

def img_to_encod(image, model):
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
        database[identity] = img_path_to_encod(file, FRmodel)
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        cv2.putText(img, std(identity), (x1+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    if identities != []:
        ready_to_detect_identity = False
        pool = Pool(processes=1) 
        pool.apply_async(welcome_users, [identities])
    return img

def find_identity(frame, x1, y1, x2, y2):
    height, width, channels = frame.shape
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    encoding = img_to_encod(image, model)
    min_dist = 100
    identity = None
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.52:
        print('unsucc threshold guess : ' + identity + ' dist : ' + str(min_dist))
        return None
    else:
        print('succ threshold guess : ' + identity + ' dist : ' + str(min_dist)+'\n\n')
        return str(identity)

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
    img = cv2.imread('G:\\pHOTOS\\Photo.jpg',1)
    frame = img
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = process_frame(img, frame, face_cascade)   
    cv2.imshow("preview", img)
    key = cv2.waitKey(0)
    cv2.destroyWindow("preview")


if __name__ == "__main__":
    elapsed_time = time.time() - start_time
    start_time=time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    windows10_voice_interface.Speak('we are up')
    database = prepare_database()
    elapsed_time = time.time() - start_time
    start_time=time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    windows10_voice_interface.Speak('we are up and runnin')
    elapsed_time = time.time() - g_start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    #webcam_face_recognizer(database)
    test_recogniser(database)


    

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
