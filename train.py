import os
import numpy as np
import cv2 as cv
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load the pre-trained FaceNet model
embedder = FaceNet()

# Initialize MTCNN detector
detector = MTCNN()

class FACELOADING:
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.target_size = (160, 160)
        self.x = []
        self.y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(img)

        if len(faces) == 0:
            raise ValueError(f"No faces found in image {filename}")

        x, y, w, h = faces[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            path = os.path.join(dir, im_name)
            if os.path.isfile(path):
                try:
                    single_face = self.extract_face(path)
                    FACES.append(single_face)
                except Exception as e:
                    print(f"Failed to process image {im_name}: {e}")
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.dataset_directory):
            path = os.path.join(self.dataset_directory, sub_dir)
            if os.path.isdir(path):
                FACES = self.load_faces(path)
                labels = [sub_dir for _ in range(len(FACES))]
                self.x.extend(FACES)
                self.y.extend(labels)
        return np.asarray(self.x), np.asarray(self.y)


# Load faces and labels
dataset_directory = 'images'
faceloading = FACELOADING(dataset_directory)
x, y = faceloading.load_classes()

if len(x) == 0 or len(y) == 0:
    print("No data loaded. Ensure the images are properly processed.")
    exit()

# Generate embeddings for all images
def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

EMBEDDED_X = np.array([get_embedding(face) for face in x])

# Save the embeddings and labels
np.savez_compressed('faces_embeddings_done_model', EMBEDDED_X, y)

# Encode labels
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

# Train SVM classifier
model = SVC(kernel='linear', probability=True)
model.fit(EMBEDDED_X, y_encoded)

# Save the trained SVM model
with open("svm_model_160x160.pkl", 'wb') as f:
    pickle.dump(model, f)

print("Model training complete and saved.")
