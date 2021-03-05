import numpy as np
import cv2
import os
from tqdm import tqdm

base_dir = './drive/MyDrive/Extract'

def detect_face(image, count):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    j = 0
    for (x, y, w, h) in faces:
        name = base_dir + "/faces/" + "face_" + str(count) + '_' + str(j) + ".jpg"
        roi_color = image[y:y + h, x:x + w]
        cv2.imwrite(name, roi_color)
        j += 1

count=0
for filename in tqdm(os.listdir(base_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(base_dir+filename)
        try:
            detect_face(image,count)
            count += 1
        except:
            pass
    else:
        continue
