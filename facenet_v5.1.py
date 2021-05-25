from multiprocessing.connection import Client
from keras import backend as K
import sys
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from subprocess import call
import io
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.svm import SVC  
from sklearn.externals import joblib
import csv
import random
import math
from cv2 import*


print('start time : ', time.strftime("%H:%M:%S", time.gmtime(time.time())))
g_start_time = start_time  = time.time()
PADDING = 0
ready_to_detect_identity = True
disp = False
CLASSIFER = False

# ========= functions that get 128d vec from server ============
def img_path_to_encod(image_path):
    img1 = cv2.imread(image_path, 1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING
        height, width, channel = img1.shape
        part_image = img1[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if 'part_image' in vars():    
        return img_to_encod(part_image)
    else:
        return img_to_encod(img1)
    
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
#--------------------------------------------------------------

# ========================= dist model =======================
def prepare_dist_model():
    model = {}
    for file in glob.glob("C:/Users/sai_charan/Desktop/Final Year Project/SangeethRaaj-fyp-e9481b120ee1/pi/facenet_v4/images/dataset/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        model[identity] = img_path_to_encod(file)
    pickle.dump( model, open( "C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//dist/emds.p", "wb" ) )
    return model
    
def load_dist_model():
    model = pickle.load( open( "C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//dist/emds.p", "rb" ) )
    return model
# ----------------------------------------------------------

# ======================= svm ===============================

def train_svm():
    embds, clss = read_from_csv('C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//main.csv')
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(embds, clss)
    joblib.dump(svclassifier, 'C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//svm//model.mdl')

def train_naive_bayes():
	dataset = loadCsv('C://Users//sai_charan//Desktop//NaiveBayes//f2.csv')
	model = summarizeByClass(dataset)
	joblib.dump(model, 'C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//nb//model.mdl')

def load_svm():
    model = joblib.load('C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//svm//model.mdl')
    return model

def load_naive_bayes():
    model = joblib.load('C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//models//nb//model.mdl')
    return model

# -----------------------------------------------------------  
#Example of Naive Bayes implemented from Scratch in Python


def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries


def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions





# ============== csv handling functions ======================
def prepare_csv():
    model = {}
    encc = []
    clss = []
    for file in glob.glob("C://Users//sai_charan//Desktop//Final Year Project//SangeethRaaj-fyp-e9481b120ee1//pi//facenet_v4//images//dataset//*"):
        identity = os.path.splitext(os.path.basename(file))[0].split('.')[0]
        print(identity)
        encc.append(img_path_to_encod(file))
        clss.append(identity)
    with open('C://Users//sai_charan//Desktop//naivebayes//f2.csv', 'w') as f:
        w = csv.writer(f)
        i = 0
        while i < len(encc):
            temp = encc[i].tolist()[0]
            if(clss[i]=='bhaskar'):
                temp.append(1)
            elif(clss[i] == 'sangeeth'):
                temp.append(2)
            else:
                temp.append(3)
            #temp.append(clss[i])
            w.writerow(temp)
            i = i+1
    return w
	



def read_from_csv(pathh):
    embds = []
    clss = []
    with open(pathh,'r',encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row) == 0):
                continue
            temp = np.array(row[:-1])
            embds.append(temp.astype(np.float).tolist())
            clss.append(row[-1])
    return embds, clss

#------------------------------------------------------------

def webcam_recogniser(model):
    cam = VideoCapture(0)   # 0 -> index of camera
    while 1 :
        s, img = cam.read()
        if s:    # frame captured without any errors
            namedWindow("preview",WINDOW_AUTOSIZE)
            imshow("preview",img)
            frame = img
            process_frame(img, frame, model)
            cv2.destroyWindow("preview")

def recogniser(model):
    print('recognition has begun')
    global ready_to_detect_identity
    global dist
    if(disp):
        cv2.namedWindow("preview")
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    stream = io.BytesIO()
    
    for framee in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = framee.array
        frame = img
        if ready_to_detect_identity:
            img = process_frame(img, frame, model)   
        key = cv2.waitKey(100)
        if(disp):
            cv2.imshow("preview", img)
        rawCapture.truncate(0)
        stream.truncate()
        stream.seek(0)
        if key == 27: # exit on ESC
            break
    if(disp):
        cv2.destroyWindow("preview")

def process_frame(img, frame, model):
    global ready_to_detect_identity
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING
        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        height, width, channel = frame.shape
        part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        identity = recognise(part_image, model)
        if identity is not None:
            identities.append(identity)
    if identities != []:
        ready_to_detect_identity = False
        pool = Pool(processes=1) 
        pool.apply_async(display, [identities])
    return img

def recognise(image, model):
    start_time  = time.time()
    encoding = img_to_encod(image)
    elapsed_time = time.time() - start_time
    
    if(CLASSIFER):
        encoding  = encoding.tolist()[0]
        identity = model.predict([encoding])[0]
        if(identity == 'unknown'):
            return None
        else:
            return identity
    else:
        min_dist = 100
        identity = None
        for (name, db_enc) in model.items():
            dist = np.linalg.norm(db_enc - encoding)
            print('distance for %s is %s' %(name, dist))
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > 0.52:
            return None
        else:
            return str(identity)

def display(identities):
    global ready_to_detect_identity
    welcome_message = 'Welcome '
    print(' in audio gen')
    if len(identities) == 1:
        welcome_message += '%s,' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += '!'
    print(welcome_message)
    cmd_beg= 'espeak '
    cmd_end= ' 2>/dev/null'
    text = welcome_message.replace(' ', '_')
    call([cmd_beg+text+cmd_end], shell=True)
    print('spoken')
    ready_to_detect_identity = True

def test_recogniser(model, path):
    img = cv2.imread(path,1)
    frame = img
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = process_frame(img, frame, model)   
    global disp
    if(disp):
        cv2.imshow("preview", img)
    key = cv2.waitKey(0)
    cv2.destroyWindow("preview")

if __name__ == "__main__":
    arg = sys.argv
    if len(arg) < 2:
        print('wrong usage')
    else:
        if arg[1] == '-t':
# ============================ train part ========================
            if arg[2] == '-svm':
                prepare_csv()
                train_svm()
            elif arg[2] == '-nb':
                prepare_csv()
                train_naive_bayes()
            elif arg[2] == '-dist':
                prepare_dist_model()
# ----------> add calls to train funtions in elif blocks here <------------
            else:
                print('wrong usage')

        elif arg[1] == '-r':
# ========================= recognise part =========================

            if(len(arg)<5):
                print ('wrong usage')
            else:
                if(arg[3] == '-d'):
                    global disp
                    disp = True
                else: 
                    global disp
                    disp = False

                if arg[2] == '-svm':
                    global CLASSIFER
                    CLASSIFER = True
                    model = load_svm()

                elif arg[2] == '-nb':
                    global CLASSIFER
                    CLASSIFER = True
                    model = load_naive_bayes()

                elif arg[2] == '-dist':
                    model = load_dist_model()

# ----------> add calls to load funtions in elif blocks here <------------

                else:
                    print('wrong usage')
                    sys.exit()
                
                if(arg[4] == '-picam'):
                    recogniser(model)
                elif arg[4] == '-load':
                    test_recogniser(model, arg[5])
                elif arg[4] == '-webcam':
                    webcam_recogniser(model)

        else:
            print('Wrong usage')


      


# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 



# arg = sys.argv[1:]
#     if len(arg) > 0:
#         if arg[0] == '-t':
#             model = prepare_model()
#             print('pickle created in ')
#             elapsed_time = time.time() - g_start_time
#             print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#         elif arg[0] == '-tc':
#             model = prepare_csv_n_model()
#             print('csv created in ')
#             elapsed_time = time.time() - g_start_time
#             print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#         elif arg[0] == '-tm':
#             clsshndlr(arg[1:])
#         elif arg[0] == '-rm':
#             clsshndlrr(arg[1:])
#         elif arg[0] == '-csv':
#             prepare_csv()
#         elif arg[0] == '-rc':
#             print('feature yet to be supported')
#             # model = load_model_csv()
#             # load_model()
#             # print('loaded the csv in')
#             # elapsed_time = time.time() - g_start_time
#             # print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#             # # recogniser(model)
#             # test_recogniser(model)
#         else:
#             model = load_model()
#             print('loaded the model in')
#             elapsed_time = time.time() - g_start_time
#             print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
#             recogniser(model)
#             #test_recogniser(model)
#     else:
#         print('Usage : python facenet_v3.py -t/-r/-tc/-rc --svm/-')
