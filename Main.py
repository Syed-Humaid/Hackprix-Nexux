from tkinter import *
from tkinter import filedialog
import cv2 as cv
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model
from SelfDriving import cannyDetection, segmentDetection, calculateLines, visualizeLines, detectPotholes, detectUnstructuredRoad, class_labels

main = Tk()
main.title("SSLA based traffic sign and lane detection for autonomous cars")
main.geometry("1300x1200")

global filename
global model

def loadModel():
    global model
    model = load_model('model/model.h5')
    pathlabel.config(text="Machine Learning Traffic Sign Detection Model Loaded")
    text.delete('1.0',END)
    text.insert(END,"Machine Learning Traffic Sign Detection Model Loaded\n\n")

def detectSignal():
    global model
    filename = filedialog.askopenfilename(initialdir="Videos")
    pathlabel.config(text=filename)
    text.delete("1.0",END)
    text.insert(END,filename+" Loaded")
    text.update_idletasks()

    cap = cv.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
           break
        canny = cannyDetection(frame)
        cv.imshow("Canny Image", canny)
        segment = segmentDetection(canny)
        hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, minLineLength=100, maxLineGap=50)
        if hough is not None:
            lines = calculateLines(frame, hough)
            linesVisualize = visualizeLines(frame, lines)
            cv.imshow("Hough Lines", linesVisualize)
            output = cv.addWeighted(frame, 0.9, linesVisualize, 1, 1)
            potholes = detectPotholes(frame)
            for (x, y, w, h) in potholes:
                cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if detectUnstructuredRoad(frame):
                cv.putText(output, "Unstructured Road", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imwrite("test.jpg", output)
            temps = cv.imread("test.jpg")
            h, w, c = temps.shape
            image = load_img("test.jpg", target_size=(80, 80))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            (boxPreds, labelPreds) = model.predict(image)
            boxPreds = boxPreds[0]
            startX = int(boxPreds[0] * w)
            startY = int(boxPreds[1] * h)
            endX = int(boxPreds[2] * w)
            endY = int(boxPreds[3] * h)
            predict = np.argmax(labelPreds, axis=1)[0]
            accuracy = np.amax(labelPreds, axis=1)
            if accuracy > 0.97:
                cv.putText(output, "Recognized As " + str(class_labels[predict]), (startX, startY), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv.imshow("Output", output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    
def close():
    main.destroy()
            
font=('times',16,'bold')
title=Label(main,text="SSLA Detection for Autonomous Cars",anchor=W, justify=CENTER)
title.config(bg='black',fg='cyan')
title.config(font=font)
title.config(height=3,width=120)
title.place(x=0,y=5)

font1=('times',14,'bold')
upload=Button(main,text="Generate & Load ML model",command=loadModel)
upload.config(font=font)
upload.place(x=50,y=100)

pathlabel=Label(main)
pathlabel.config(bg='black',fg='cyan')
pathlabel.config(font=font)
pathlabel.place(x=50,y=150)

markovButton=Button(main,text="Upload Video & Detect Hough Lane,Signal",command=detectSignal)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

predictButton=Button(main,text="Exit",command=close)
predictButton.place(x=5,y=250)
predictButton.config(font=font1)

font1=('times',12,'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg='#696773')
main.mainloop()