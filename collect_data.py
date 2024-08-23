import cv2
import os

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if facedetect.empty():
    print("Error: Haar Cascade XML file not found.")
    video.release()
    cv2.destroyAllWindows()
    exit()

count = 0

# Prompt user for name
nameID = input("Enter Your Name: ").strip().lower()
path = f'images/{nameID}'

# Check if directory exists
if os.path.exists(path):
    print("Name Already Taken")
    nameID = input("Enter Your Name Again: ").strip().lower()
    path = f'images/{nameID}'
    if os.path.exists(path):
        print("Name still taken. Exiting...")
        video.release()
        cv2.destroyAllWindows()
        exit()
else:
    os.makedirs(path)

# Save the nameID in a text file
with open('data/username.txt', 'w') as f:
    f.write(nameID)

print(f"Collecting images for {nameID}...")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect faces
    faces = facedetect.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        count += 1
        name = f'{path}/{count}.jpg'
        print(f"Creating image: {name}")
        cv2.imwrite(name, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow("WindowFrame", frame)

    # Exit condition: Stop if more than 20 images are collected
    if count >= 200:
        print("Image collection complete.")
        break

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
