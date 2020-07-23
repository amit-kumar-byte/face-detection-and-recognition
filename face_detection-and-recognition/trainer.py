import cv2, threading, os
from PIL import Image
import numpy as np


haarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

def labels_for_training_data(path):
    faceSamples = []
    faceID = []

    img_paths = [os.path.join(path, f) for f in os.listdir(path)]

    for imgPath in img_paths:
        pilImg = Image.open(imgPath).convert('L')
        imgNp = np.array(pilImg, 'uint8')
        Id = int(os.path.split(imgPath)[-1].split("-")[1])

        faces = haarCascade.detectMultiScale(imgNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imgNp[y:y+h, x:x+h])
            faceID.append(Id)
        
    
    
    return faceSamples, faceID


def train():
    stopEvent = threading.Event()
    faces, ids = labels_for_training_data('dataset')
    recognizer.train(faces, np.array(ids))
    print('Successfully trained')
    recognizer.save('training.yml')
    stopEvent.set()



    
