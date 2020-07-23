import tkinter as tk
import threading
from FaceDetection import capture
from trainer import train
from recognizer import recognize

root = tk.Tk()
frame1 = tk.Frame(root)
frame1.pack(fill=tk.BOTH, expand=True)

frame2 = tk.Frame(root)
frame2.pack(fill=tk.BOTH, expand=True)

root.configure(bg='white')

def getData():
    threading.Thread(target=capture).start()

def trainData():
    threading.Thread(target=train).start()

def recognition():
    threading.Thread(target=recognize).start()

def exit():
    root.destroy()

root.title("Face Detection & Recognition")

tk.Label(frame1, text = "Face Detection & Recognition", font=("times new roman",25),bg = '#098043', fg='white'  ).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Button(frame2, text="Capture Dataset", font=('times new roman', 18),bg='#098043', fg='white' , command=getData).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Button(frame2, text="Train Dataset", font=('times new roman', 18),bg='#098043', fg='white' , command=trainData).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Button(frame2, text="Recognize", font=('times new roman', 18),bg='#098043', fg='white' , command=recognition).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tk.Button(root, text="Exit", font=('times new roman', 18),bg='#098043', fg='white', command=exit ).pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()

