import tkinter as tk
from tkinter import filedialog
from tkinter import *  
import os
import subprocess
import numpy 
 
#initialise GUI
top=tk.Tk()
top.geometry('1200x720')
top.title('Moving Object Detection using Frame Differencing')
bg = PhotoImage(file = "a.png")
canvas1 = Canvas( top, width = 800, height = 800)

canvas1.pack(fill = "both", expand = True)

# Display image
canvas1.create_image( 0, 0, image = bg, anchor = "nw")



# top.configure(background= bg)

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path): 
    # print(file_path)
    # S=file_path.split("/")
    # L=len(S)-1
    # Str="detect.py --input input/"+S[L]+" -c 4"
    Str="detect.py --input "+file_path+" -c 4"
    print(Str)
    # harActivity = "Main1.py --input inputVideos/bridge.mp4 --output outputVideos/bridgeOut.avi --yolo yolo-"
    subprocess.call("python "+Str)
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Get Moving Object Detection",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
    button2_canvas = canvas1.create_window( 800, 300, anchor = "nw", window = classify_b)

def upload_video():
    try:
        file_path=filedialog.askopenfilename() 
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

def Hand_Finger(): 
    Str="Handweb.py" 
    subprocess.call("python "+Str) 


upload=Button(top,text="Moving Object Upload A Video",command=upload_video,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
button1_canvas = canvas1.create_window( 100, 500, anchor = "nw", window = upload)


Handupload=Button(top,text="Hand,Finger Counting Detection",command=Hand_Finger,padx=10,pady=5)
Handupload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

Handupload.pack(side=BOTTOM,pady=50)
button1_canvas = canvas1.create_window( 450, 500, anchor = "nw", window = Handupload)

sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Moving Object and Hand,Finger Counting Detection using Frame Differencing",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#FF0000')
heading.pack()
button2_canvas = canvas1.create_window( 50, 100, anchor = "nw", window = heading)

top.mainloop()
