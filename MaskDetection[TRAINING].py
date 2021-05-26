from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import cv2

with_mask = np.load('with_mask.npy', allow_pickle=True)
without_mask = np.load('without_mask.npy', allow_pickle=True)

with_mask = with_mask.reshape(with_mask.shape[0], 50 * 50 * 3)
without_mask = without_mask.reshape(without_mask.shape[0], 50 * 50 * 3)

X = np.r_[with_mask, without_mask]

labels = np.zeros(X.shape[0])
labels[with_mask.shape[0]:] = 1.0
names = {0: 'Mask', 1: 'No Mask'}
color = {0: (255, 255, 0), 1: (0, 255, 255)}

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

pca = PCA(n_components=3)
X_train = pca.fit_transform(x_train)

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

svm = SVC()
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

haar_data = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(
            img)
        for x, y, w, h in faces:
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            pred = svm.predict(face)
            n = names[int(pred)]
            colour = color[int(pred)]
            cv2.rectangle(img, (x, y), (x+w, y+h), colour, 4)
            cv2.putText(img, n, (x, y), font, 1, colour, 2)
            print(n)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27 or len(data) >= 200 and 0xFF == ord('q'):
            break
capture.release()
cv2.destroyAllWindows()
