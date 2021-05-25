A wearable and portable prototype that can be worn and carried around with ease was designed to help the Visually Impaired people recognize the faces of known people in their presence. This prototype has an embedded Raspberry Pi and uses a Pi camera which aids the user by detecting, capturing, and recognizing faces through several techniques and then providing an output in the form of an audio message.

# facenet-face-recognition

This repository contains a demonstration of face recognition using the FaceNet network (https://arxiv.org/pdf/1503.03832.pdf) and a webcam. Our implementation feeds frames from the webcam to the network to determine whether or not the frame contains an individual we recognize.

## How to use

To install all the requirements for the project run

	pip install -r requirements.txt

In the root directory. After the modules have been installed you can run the project by using python

	python facenet.py

## NOTE

We are using the Windows 10 Text-to-Speech library to output our audio message, so if you want to run it on a different OS then you will have to replace the speech library used in facenet.py


## Prototype
![a](https://user-images.githubusercontent.com/22254732/119573290-a745e400-bd79-11eb-96ad-503169ba5530.png)
