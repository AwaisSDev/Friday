import cv2
import pickle

def recognize_faces(threshold=100):  # Set threshold to 60
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.yml')
    
    # Load the label dictionary
    with open('label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    
    print("Label dictionary loaded:", label_dict)  # Print to verify
    
    video_capture = cv2.VideoCapture(1)
    
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            t = 200
            face_image = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_image)
            
            if confidence < threshold:  # Use the threshold to decide if the recognition is valid
                name = label_dict.get(label, "Unknown")
            elif confidence > t:
                name = "Unknown"
            
            (f"Detected Face - Label: {label}, Name: {name}, Confidence: {confidence}")  # Debug print
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
recognize_faces(threshold=100)  # Set the confidence threshold to 60
