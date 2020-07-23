import cv2, time, threading, os
import pandas as pd


def getName(id):
    if not os.path.exists('data.csv'):
        return None
    
    df1 = pd.read_csv('data.csv' )

    df1.set_index('Id', inplace=True)
    return df1.loc[id, 'Name']



def recognize():
    stopEvent = threading.Event()
    start = time.time()
    period = 20

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    haarCascade = cv2.CascadeClassifier('haarCascade/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognizer.read('training.yml')

    
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haarCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]

            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            id, conf = recognizer.predict(roi_gray)

            name = getName(id)

            cv2.putText(img, str(name) + ' ' + str(conf), (x,y-10), font, 0.55, (120, 255,120),1)
        
        cv2.imshow('frame', img)
        
        if time.time() > start+period:
            break

        elif cv2.waitKey(100) & 0xff == ord('q'):
            break
    
    cap.release()
    stopEvent.set()
    cv2.destroyAllWindows()




