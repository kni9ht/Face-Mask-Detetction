import cv2
import numpy as np
import glob

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

data1 = []
files = glob.glob("dataset/with_mask/*.jpg") + \
    glob.glob("dataset/with_mask/*.jpeg")
for myFile in files:
    img = cv2.imread(myFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = img[y: y + h, x: x + w]
        face = cv2.resize(face, (50, 50))
        if img is not None:
            data1.append(face)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()

np.save('with_mask.npy', data1)

data2 = []
files = glob.glob("dataset/without_mask/*.jpg")+glob.glob(
    "dataset/without_mask/*.PNG")+glob.glob("dataset/without_mask/*.jpeg")
for myFile in files:
    img = cv2.imread(myFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = img[y: y + h, x: x + w]
        face = cv2.resize(face, (50, 50))
        if img is not None:
            data2.append(face)
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()

np.save('without_mask.npy', data2)
