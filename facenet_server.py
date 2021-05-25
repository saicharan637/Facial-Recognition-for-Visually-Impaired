from multiprocessing.connection import Listener
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

if __name__ == "__main__":

    handler(sys.argv[1:])

    elapsed_time = time.time() - start_time
    start_time=time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    windows10_voice_interface.Speak('we are up')
    address = ('', 6000)
    listener = Listener(address, authkey=str.encode('secret password'))
    windows10_voice_interface.Speak('we are up and runnin')
    print('helloooo....')
    conn = listener.accept()
    print ('connection accepted from', listener.last_accepted)
    while True:
        msg = conn.recv()
        if(msg == 'now'):
            msg = conn.recv()
            x_train = np.array([msg])
            encc = FRmodel.predict_on_batch(x_train)
            conn.send('encc')
            conn.send(encc)
            conn.send('close')
        else:
            if(msg == 'close'):
                listener.close()
                listener = Listener(address, authkey=str.encode('secret password'))
                conn = listener.accept()


    

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
