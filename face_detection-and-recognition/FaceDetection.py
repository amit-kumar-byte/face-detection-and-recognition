import cv2, os
import pandas as pd
import tkinter.messagebox 
import tkinter as tk
import threading




def check_path(path):
    
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    

def store(id, name):
    if not os.path.exists('data.csv'):
        df = pd.DataFrame(columns=['Id', 'Name'])
        df.to_csv('data.csv', index=False)

    df1 = pd.read_csv('data.csv' )
   

    data = [id, name]
    ab = df1[df1['Id']== id]

    
    if ab.empty:        
        df1 = df1.append(pd.DataFrame([[id, name]], columns=['Id', 'Name']))
        df1.to_csv('data.csv', index=False)
        return True
            
    else:
        root = tk.Tk()
        #root.withdraw()
        tkinter.messagebox.showwarning("Message", "Id already exists!")
        root.destroy()

           


 
def capture():

    stopEvent = threading.Event()

    while True:
        id = int(input("Enter your ID: "))
        name = input("enter your name: ")
        
        if store(id, name):
            break
        else:
            continue
        
        

    # start capturing video

    vid_cam = cv2.VideoCapture(0)

    # Detect object in video stream using haarcascade frontal face
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #Initialize sample face image
    count = 0
    
    check_path("dataset/")

    #start looping
    while(True):

        #Capture video frame
        ret, image_frame = vid_cam.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Loops for each faces
        for (x,y,w,h) in faces:

            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x,y), (x+w, y+h), (9,128,67), 2)

            roi_gray = gray[y:y+h, x:x+w]


            # Increment sample face image
            count += 1 
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User-"+ str(id) + "-" + name + "-" + str(count)+".jpg", gray[y:y+h, x:x+w])
            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('frame', image_frame)

        # To stop taking video, press 'q' for at least 100 ms
        if cv2.waitKey(100) & 0xFF == ord('q'):
            #stopEvent.set()
            break
    
        # If image taken reach 100, stop taking video
        elif count>=100:
            print("Successfully captured")
            break

    # Stop video
    vid_cam.release()
    stopEvent.set()
    # Close all started windows
    cv2.destroyAllWindows()




