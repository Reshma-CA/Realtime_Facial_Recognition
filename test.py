
import cv2 as cv
import numpy as np
import os
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle

# Load FaceNet and SVM model
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_model.npz")
Y = faces_embeddings['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)

haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Open camera
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160))  # Resize to FaceNet input size
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)  # Get embeddings

        face_name = model.predict(ypred)  # Predict with SVM
        final_name = encoder.inverse_transform(face_name)[0]  # Decode prediction

        # Draw rectangle and name
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(frame, final_name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()