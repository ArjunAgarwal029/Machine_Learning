import cv2
from keras.models import model_from_json
import numpy as np

f= open("emotiondectector.json","r")
model_json=f.read()
f.close()
model=model_from_json(model_json)

model.load_weights("emotiondectector.h5")
face_cascade=cv2.CascadeClassifier("./opencv/haarcascade_frontalface_default.xml")

def extract_features(image):
    feature =np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

webcam=cv2.VideoCapture(0)

labels={0 : 'angry',1 : 'disgusted',2 : 'fearful',3 : 'happy',4 : 'neutral',5 : 'sad',6 : 'surprised'}
while True:
    i,im=webcam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    try:
        for (x,y,w,h) in faces:
            image= gray[y:y+h,x:x+w]
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img =extract_features(image)
            pred = model.predict(img)
            predicition_label=labels[pred.argmax()]
            cv2.putText(im,'% s'%(predicition_label),(x-10, y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255))
        cv2.imshow("Output",im)
        cv2.waitKey(27)

    except cv2.error:
        pass