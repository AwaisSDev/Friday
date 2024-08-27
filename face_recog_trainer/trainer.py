import cv2
import numpy as np
import os
import pickle

def train_model():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_dict = {}
    label_counter = 0
    
    # Load the dataset
    for user_folder in os.listdir('dataset'):
        user_path = f'dataset/{user_folder}'
        if not os.path.isdir(user_path):
            continue
        
        label_dict[label_counter] = user_folder
        print(f"Label {label_counter} assigned to {user_folder}")
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces_detected:
                faces.append(image[y:y+h, x:x+w])
                labels.append(label_counter)
        
        label_counter += 1
    
    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    recognizer.save('trained_model.yml')
    
    # Save the label dictionary
    with open('label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    
    print("Model trained and saved as 'trained_model.yml'")
    print("Label dictionary saved as 'label_dict.pkl'")

# Example usage
train_model()
