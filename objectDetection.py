import numpy as np 
import cv2

#all category list 
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]

#random frame colors
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))

#load model
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")

def Detection(cap):
    while True:
        #read each frame
        ret, frame = cap.read()
        if ret:
            (h,w) = frame.shape[:2]
            #preprocessing
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
            net.setInput(blob)
            #feed to model and keep result in "detections"
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                percent = detections[0,0,i,2]
                #filter more than 0.5
                if percent > 0.5:
                    class_index = int(detections[0,0,i,1])
                    box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                    (startX, startY, endX, endY) = box.astype("int")

                    #aesthetic/frame and name
                    label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                    cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                    y = startY - 15 if startY-15>15 else startY+15
                    cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
                
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    #clear memory and close webcam
    cap.release()
    cv2.destroyAllWindows()

def StaticDetection(cap):
    ret, frame = cap.read()
    if ret:
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0,0,i,2]
            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                y = startY - 15 if startY-15>15 else startY+15
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
        while True:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

#video or image or webcam 
source = input("1.video 2.image 3.webcam(type number): ")

if source == "1":
    source = input("filename: ") 
    cap = cv2.VideoCapture(source)
    Detection(cap)
if source == "2":
    source = input("filename: ") 
    cap = cv2.VideoCapture(source)
    StaticDetection(cap)
if source == 3:
    cap = cv2.VideoCapture(0)
    Detection(cap)
        

