### Real time Face Recognition Project
This face recognition project leverages advanced computer vision and machine learning techniques to accurately identify individuals from live video feeds. The system utilizes the FaceNet model from the keras-facenet library to extract facial embeddings, which are then used for recognition. The MTCNN library is employed for face detection, ensuring precise localization of facial features within images. The project also integrates an SVM classifier, trained on a custom dataset of known faces, to perform the actual recognition. Unknown faces are stored separately for further analysis and model improvement. The entire system is designed to handle real-time face recognition from webcam input, with a user-friendly interface displaying identified names and drawing bounding boxes around detected faces. This project demonstrates practical applications of deep learning in face recognition, showcasing both the effectiveness and real-world usability of modern machine learning techniques.





### step.1. Image Collection for Face Recognition

To build a robust face recognition model, you need to collect facial images for each individual you want to recognize. This process involves capturing images using a webcam and associating them with a user name. Follow these steps to collect images:

#### 1. Run the Image Collection Script:

Launch the collect_data.py script to start capturing images through your webcam:

```bash
python collect_data.py

```
#### 2.Enter User Name:
When prompted, enter the user name for the person whose images you are collecting. This name will be used to label the images and associate them with the correct individual in the dataset.

#### 3.Image Capture:

The script will start capturing images from the webcam. As it detects faces, it will save the images in a directory named after the entered user name (inside the images/ directory). The script will collect images until you reach the desired number, typically around 200 images, to ensure good model performance.

#### 4. Handling Existing Names:

If the entered user name already exists in the dataset directory, you will be prompted to enter a new name. Make sure to provide a unique name to avoid overwriting existing data.

#### 4. Exit the Script:

To stop the image collection process, you can press 'q' on your keyboard. The script will then save the collected images and exit.

This step is crucial for creating a well-trained face recognition model, as the quality and quantity of the collected images directly impact the model's accuracy.

### step2.Face Recognition Model Training

Once you have collected a sufficient number of images for each individual, the next step is to train the face recognition model. This involves processing the collected images to generate embeddings, training a classifier, and saving the trained model for future use. Follow these steps for training the face recognition model:

#### 1.Run the Training Script:

Execute the train.py script to start the training process:

```bash
python train.py

```
#### 2.Processing Images:

The script will load the collected images from the images/ directory, extract facial embeddings using the FaceNet model, and prepare the data for training. The embeddings are vector representations of the faces, which help in distinguishing between different individuals.

#### 3.Train the Classifier:

Using the embeddings, the script trains an SVM (Support Vector Machine) classifier. The classifier learns to associate each embedding with a specific user name, allowing it to recognize individuals based on their facial features.

#### 4.Save the Model:

After training, the script saves the embeddings and the trained SVM model to files (faces_embeddings_done_model.npz and svm_model_160x160.pkl, respectively). These files are essential for the face recognition process during testing and deployment.

#### 5.Verify Training Completion:

Once the training script finishes, you should see a message indicating that the model training is complete and the model has been saved successfully. Ensure that no errors occurred during training.

This step is vital for creating a face recognition system that can accurately identify individuals based on the images you collected. Proper training ensures that the model can generalize well to new images of the same individuals.

### step 3. Face Recognition Model Testing
This script implements a real-time face recognition system using a webcam. The process is designed to be straightforward and includes several key steps:

#### 1.Load Pre-trained Models:

FaceNet Model: Utilized to generate face embeddings, which are essential for face recognition.
SVM Classifier: Trained on these embeddings to classify faces into known identities.
Label Encoder: Converts the numerical labels from the SVM model back into human-readable names.

#### 2. Setup Webcam Capture:

Initializes the webcam to start capturing video frames for processing.

```bash
python test.py

```
#### 3.Real-Time Face Detection and Recognition:

Face Detection: Uses Haar Cascade to detect faces within each frame from the webcam feed.
Face Extraction: Crops and resizes detected faces to 160x160 pixels, suitable for processing by FaceNet.
Face Embedding: Computes face embeddings using FaceNet for each detected face.
Face Classification: Predicts the identity of the face using the SVM classifier and decodes the prediction to a name using the Label Encoder.
#### 4.Display and Visualization:

Draws rectangles around detected faces and displays the predicted names directly on the video feed.
Continuously updates the display to show real-time recognition results.
#### 5.Exit the Application:

Closes the video feed and releases resources when the 'q' key is pressed.
Make sure that the required files (faces_embeddings_done_model.npz, svm_model_160x160.pkl, and haarcascade_frontalface_default.xml) are correctly placed in the working directory for the script to function correctly.

## Project Structure & setup

collect_data.py: Script for collecting images of faces and saving them in a dataset directory.
train.py: Script for training a face recognition model using collected images.
test.py: Script for testing the trained model by recognizing faces using a webcam.
my_venv/: Virtual environment directory (ignored by Git).

### 1.Installation
To set up and run this project, follow these steps:

Clone the Repository:


```bash
git clone https://github.com/Reshma-CA/Face_Image_Recognition_Real_timeproject.git
cd Face_Recognition_project

```

### 2.Create a Virtual Environment:

Ensure you have Python installed. Create a virtual environment using venv:

```bash
python -m venv my_venv

```
### 3.Activate the Virtual Environment:
```bash
my_venv\Scripts\activate

```
### 4.Install Required Packages:

Install the required Python packages using requirements.txt:

```bash
pip install -r requirements.txt

```
## Usage
#### 1. Collect Data:

Run the collect_data.py script to collect images for training:

```bash
python collect_data.py

```

#### 2.Train the Model:

After collecting data, train the model with the train.py script:


```bash
python train.py

```
#### 3.Test the Model:

Test the trained model with the test.py script:

```bash
python test.py

```

## Notes
Ensure the Haar Cascade XML file (haarcascade_frontalface_default.xml) is in the project directory.
The virtual environment directory (my_venv/) is ignored by Git to avoid committing unnecessary files.
