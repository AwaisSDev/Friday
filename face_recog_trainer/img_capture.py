import cv2
import os

def capture_images(name, num_images=100):
    # Create a directory to save images if it doesn't exist
    os.makedirs(f'dataset/{name}', exist_ok=True)
    
    # Load the Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize the webcam
    video_capture = cv2.VideoCapture(1)
    
    count = 0
    while count < num_images:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Save images of detected faces
        for (x, y, w, h) in faces:
            count += 1
            face_image = gray[y:y+h, x:x+w]
            cv2.imwrite(f'dataset/{name}/user_{count}.jpg', face_image)
            cv2.imshow('Capturing Images', face_image)
        
        # Display the frame
        cv2.imshow('Video', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    video_capture.release()
    cv2.destroyAllWindows()
    
    print(f"Captured {count} images for {name}")

# Example usage
user_name = input("Enter your name: ")
capture_images(user_name)
