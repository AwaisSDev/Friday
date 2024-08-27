import sys
import pyttsx3
import speech_recognition as sr
import os
import webbrowser
import datetime
import random
import uuid
import smtplib
import cv2
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq
from PyQt5 import QtCore, QtGui, QtWidgets
import queue
from github import Github
import pywhatkit as kit
import time
import pyautogui
import imaplib
import email
from email.header import decode_header
import pickle
import threading

# Initialize the recognizer and text-to-speech engine
recognizer = sr.Recognizer()
MICROPHONE_INDEX = 1
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Create a queue for TTS requests
tts_queue = queue.Queue()

mode = "text"  # Set default mode to text
bypass_words = ["!", "$", "^", "&", "*", "/", "asteras", "😊"]
known_face_encodings = []
known_face_names = []

# Global flag to control the hand mouse operation
hand_mouse_active = False
hand_mouse_thread = None  # To store the thread reference

# Function to stop the hand mouse control
def stop_hand_mouse():
    global hand_mouse_active
    hand_mouse_active = False

history = []  # Initialize history

github_token = os.getenv("GITHUB_TOKEN")
friday_folder = "h:/New Coding/JARVIS_IM/Friday"
repo_name = "AwaisSDev/MadebyFriday"

if not github_token:
    raise ValueError("GitHub token is not set in the environment variables")

def say(audio, speed_adjustment=3): 
    # Handle specific replacements for pronunciation
    replacements = {
        "Awais": "ah-WASS",  # Pronunciation replacement
        # Add more replacements as needed
    }

    for old_word, new_word in replacements.items():
        audio = audio.replace(old_word, new_word)

    # Split the audio text into sentences and process each sentence
    sentences = audio.split(". ")
    for sentence in sentences:
        # Split the sentence into words and filter out bypass words
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in bypass_words]
        filtered_sentence = " ".join(filtered_words)

        # Add the filtered sentence to the TTS queue if it's not empty
        if filtered_sentence:
            tts_queue.put((filtered_sentence, speed_adjustment))
        else:
            print("")

    # Process TTS queue
    while not tts_queue.empty():
        text, speed_adjustment = tts_queue.get()
        rate = engine.getProperty('rate')
        new_rate = rate + speed_adjustment
        engine.setProperty('rate', new_rate)
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

def listen():
    with sr.Microphone(device_index=MICROPHONE_INDEX) as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio, language='en-pk').lower() 
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
        return ""

def send_email(subject, body, to_email):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    from_email = 'mawais9171@gmail.com'
    password = 'ipau ainb zjjt ajmk'

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        print('Email sent successfully!')
        say('Email sent successfully!')
    except Exception as e:
        print(f'Error: {e}')
        say(f'Error: {e}')
    finally:
        server.quit()

def recognize_faces(threshold=100):  # Set threshold to 100
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trained_model.yml')
    
    # Load the label dictionary
    with open('label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    
    #print("Label dictionary loaded:", label_dict)  # Print to verify
    
    video_capture = cv2.VideoCapture(1)
    
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_image = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_image)
            
            if confidence < threshold:  # Face matches
                name = label_dict.get(label, "Unknown")
                print(f"Welcome {name}, Entering Friday")
                say(f"Welcome {name}, Entering Friday")
                video_capture.release()
                cv2.destroyAllWindows()
                # Trigger "Friday" functionality here
                return "Entered Friday"
            else:  # Face does not match
                print("Access denied due to unknown face detected")
                say("Access denied due to unknown face detected")
                
                # Prompt user for PIN
                say("Please enter your PIN to gain access.")
                user_pin = input("Enter PIN: ")

                # Hardcoded PIN for simplicity (you can change this)
                correct_pin = os.getenv("FridayKey")
                print(correct_pin)
                
                if user_pin == correct_pin:
                    print("PIN verified successfully. Access granted.")
                    say("PIN verified successfully. Access granted.")
                    video_capture.release()
                    cv2.destroyAllWindows()
                    # Trigger "Friday" functionality here
                    return "Entered Friday with PIN"
                else:
                    print("Incorrect PIN. Access denied.")
                    say("Incorrect PIN. Access denied.")
                    video_capture.release()
                    cv2.destroyAllWindows()
                    sys.exit()  # Exits the program
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

def create_program_file(response_text, file_name, description):
    os.makedirs(friday_folder, exist_ok=True)
    code = extract_code(response_text)
    description_comment = f'# Project Description: {description}\n# Write code only\n\n'
    full_code = description_comment + code
    file_path = os.path.join(friday_folder, f"{file_name}.py")
    
    # Write or overwrite the file
    with open(file_path, "w") as file:
        file.write(full_code)
    
    print(f"Program created/updated: {file_path}")
    return file_path

def upload_to_github(file_path, repo_name):
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        raise ValueError("GitHub token is not set in the environment variables")

    # Initialize GitHub client
    g = Github(github_token)
    user = g.get_user()
    repo = user.get_repo(repo_name)
    
    # Get the file name from the file path
    file_name = os.path.basename(file_path)
    
    # Read the file content
    with open(file_path, "r") as file:
        content = file.read()
    
    # Upload the file to GitHub
    repo.create_file(file_name, "Adding new project file", content)
    print(f"Project has been stored to GitHub: {file_name}")

def update_github_file(file_path, repo_name):
    repo_name = "Madebyfriday"

    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        raise ValueError("GitHub token is not set in the environment variables")

    # Initialize GitHub client
    g = Github(github_token)
    user = g.get_user()

    # Ensure the repo_name is in the format 'username/repository'
    try:
        repo = user.get_repo(repo_name)
    except Exception as e:
        print(f"Repository not found: {e}")
        raise ValueError(f"Repository {repo_name} not found on GitHub")

    # Get the file name from the file path
    file_name = os.path.basename(file_path)
    
    # Read the file content
    with open(file_path, "r") as file:
        content = file.read()

    # Get the file's SHA for updating
    try:
        file_info = repo.get_contents(file_name)
        sha = file_info.sha
    except Exception as e:
        print(f"File not found on GitHub: {e}")
        raise FileNotFoundError(f"File {file_name} not found on GitHub")

    # Update the file on GitHub
    repo.update_file(file_name, "Updating project file", content, sha)
    print(f"Project has been updated on GitHub: {file_name}")

def extract_code(response_text):
    print(response_text)
    lines = response_text.split("\n")
    code_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            code_lines.append(line)
    return "\n".join(code_lines)

contacts = {
    #example
    'Abc': '+xxxxxxxxxxxxx',
}

def get_phone_number(name):
    return contacts.get(name.lower(), None)

def extract_contact_name(query):
    # Implement logic to extract contact name from the query
    # For simplicity, assume the contact name is always at the start of the query
    parts = query.split(" ", 2)
    if len(parts) > 1:
        return parts[1]  # Adjust this based on how the name is structured in the query
    return ""

def extract_message(query):
    # Implement logic to extract message from the query
    # Assume message starts after the contact name
    parts = query.split(" ", 2)
    if len(parts) > 2:
        return parts[2]
    return ""

def process_query(query, contacts):
    import os
    global mode
    if not query.strip():  # Check if query is empty or just whitespace
        say("The query cannot be empty. Please provide a valid command.")
        return ""

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key is None:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)
    model = "mixtral-8x7b-32768"

    # Check the context for relevant previous queries
    relevant_context = ""
    if history:
        last_query = history[-1]['query']
        last_response = history[-1]['response']
        relevant_context = f"Earlier you asked about '{last_query}', and I responded with '{last_response}'. "

    if "send email" in query.lower():
        say("Who is the recipient of the email?")
        if mode == "listening":
            to_email = listen()
        else:
            to_email = input("Enter the recipient email address: ")

        if not to_email:  # Check if to_email is empty
            say("Recipient email not provided. Please try again.")
            return ""

        say("What is the subject of the email?")
        subject = listen() if mode == "listening" else input("Enter the subject of the email: ")

        if not subject:  # Check if subject is empty
            say("Email subject not provided. Please try again.")
            return ""

        say("What is the body of the email?")
        body = listen() if mode == "listening" else input("Enter the body of the email: ")


        if not body:  # Check if body is empty
            say("Email body not provided. Please try again.")
            return ""

        if to_email and subject and body:
            send_email(subject, body, to_email)
        else:
            say("Sorry, I didn't catch that. Please try again.")
        return ""
    
    elif "check email" in query:
        email_count = check_email("youremail.com", "yourapppass")
        if email_count is not None:
            response = f"You have {email_count} emails in your inbox."
        else:
            response = "I couldn't check your emails right now."
        return response
    elif "create a project" in query.lower():
        say("Sure sir, should I store it to your GitHub, sir?")
        response = input("Should I store it to your GitHub? (yes/no): ")
        
        store_to_github = response.lower() in ["yes", "y", "sure"]

        say("Please provide a description for the project.")
        description = input("Enter the project description: ")

        if description:
            say("Please provide a name for the project file.")
            file_name = input("Enter the project file name: ")
            file_name = file_name.replace(" ", "_")  # Replace spaces with underscores

            prompt = f"Generate a Python project that {description}"
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                )
                response_text = chat_completion.choices[0].message.content
                file_path = create_program_file(response_text, file_name, description)

                if store_to_github:
                    say("Storing the project to your GitHub repository.")
                    upload_to_github(file_path, "Madebyfriday")
                    print("Project has been stored to GitHub.")

                say(f"Project {file_name} created successfully.")

            except Exception as e:
                print(f"An error occurred: {e}")
                say("An error occurred while creating the project.")
        return ""

    elif "make a image" in query.lower():
        import os
        import requests

        def generate_image_via_api(prompt, filename):
            # URL for the stabilityai/stable-fast-3d model
            API_URL = "https://api-inference.huggingface.co/models/goofyai/3d_render_style_xl"
            headers = {
                "Authorization": "Bearer yourapikey"
            }
            payload = {
                "inputs": prompt
            }
            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()  # Raise an error for HTTP error responses
                if response.status_code == 200:
                    # Ensure the directory exists
                    os.makedirs("friday/ai_images", exist_ok=True)
                    # Save the image in the specified folder
                    with open(os.path.join("friday/ai_images", filename), "wb") as f:
                        f.write(response.content)
                    print(f"Image saved as ai_images/{filename}")
                else:
                    print(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")

        prompt = input("Enter your prompt: ")
        imgname = input("Enter the name of the image file (e.g., 'my_image'): ")
        filename = imgname + ".png"  # Combine the input name with ".png" extension

        generate_image_via_api(prompt, filename)

    elif "hand mouse" in query.lower():
        import cv2
        import mediapipe as mp
        import pyautogui

        # Initialize MediaPipe Hands and drawing utils
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        # Initialize the hand detector
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

        # Smoothing parameters
        smoothing_factor = 0.2
        prev_x, prev_y = 0, 0  # Initialize previous coordinates for smoothing

        # Movement threshold to prevent unintended closures
        movement_threshold = 5  # Pixels

        # Function to check if two fingers are aligned
        def are_fingers_aligned(finger1, finger2, threshold=0.05):
            return abs(finger1.x - finger2.x) < threshold and abs(finger1.y - finger2.y) < threshold

        # Function to capture and process video in the background
        def video_capture_thread():
            global prev_x, prev_y, hand_mouse_active  # Use global variables inside the thread

            cap = cv2.VideoCapture(1)

            # Get the screen size
            screen_width, screen_height = pyautogui.size()

            while cap.isOpened() and hand_mouse_active:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)

                # Resize the frame to match the screen resolution
                frame = cv2.resize(frame, (screen_width, screen_height))

                # Convert the frame to RGB for MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                # Check if hand landmarks are detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Get landmark coordinates for index and middle finger tips
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                        # Convert normalized coordinates to pixel coordinates
                        index_tip_px = (int(index_tip.x * screen_width), int(index_tip.y * screen_height))
                        middle_tip_px = (int(middle_tip.x * screen_width), int(middle_tip.y * screen_height))

                        # Smooth the mouse movement
                        smooth_x = int(prev_x * (1 - smoothing_factor) + index_tip_px[0] * smoothing_factor)
                        smooth_y = int(prev_y * (1 - smoothing_factor) + index_tip_px[1] * smoothing_factor)

                        # Calculate the movement distance
                        movement_distance = ((smooth_x - prev_x) ** 2 + (smooth_y - prev_y) ** 2) ** 0.5

                        # Only move the mouse if the movement is significant
                        if movement_distance > movement_threshold:
                            pyautogui.moveTo(smooth_x, smooth_y)

                        # Update previous position
                        prev_x, prev_y = smooth_x, smooth_y

                        # Check if index and middle fingers are aligned
                        if are_fingers_aligned(index_tip, middle_tip):
                            pyautogui.click()

            # Release resources
            cap.release()

        # Start the video capture in a background thread
        hand_mouse_active = True
        hand_mouse_thread = threading.Thread(target=video_capture_thread)
        hand_mouse_thread.daemon = True
        hand_mouse_thread.start()

        print("Hand mouse control is running in the background...")
        return "Hand mouse control started. You can now use your hand gestures to control the mouse."

    elif "normal mouse" in query.lower():
        hand_mouse_active = False
        hand_mouse_thread = None  # To store the thread reference
        if hand_mouse_active:
            stop_hand_mouse()
            if hand_mouse_thread:
                hand_mouse_thread.join()  # Ensure the thread is fully stopped
            print("Switched back to normal mouse control.")
            return "Hand mouse control stopped. You are now using the normal mouse."
        else:
            return "Hand mouse control was not active."

    elif "update a project" in query.lower():
            say("Please provide the name of the project file to update.")
            file_name = input("Enter the project file name: ").replace(" ", "_")
            file_path = os.path.join(friday_folder, f"{file_name}.py")
            
            if not os.path.exists(file_path):
                say("File does not exist. Please check the file name and try again.")
                return

            say("Please provide a new description for the project.")
            description = input("Enter the new project description: ")

            if description:
                prompt = f"Generate Python code that matches the description: {description}"
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                    )
                    response_text = chat_completion.choices[0].message.content
                    create_program_file(response_text, file_name, description)
                    update_github_file(file_path, "your-repository-name")
                    say(f"Project {file_name} updated successfully.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    say("An error occurred while updating the project.")

    else:
        try:
            # Construct the prompt with relevant context
            prompt = f"{relevant_context}{query}"
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
            )
            
            # Extract the response text
            response_text = chat_completion.choices[0].message.content
            MAX_LENGTH = 30000  # Maximum length for the response
            
            # Define word replacements
            replacements = {
                "trained by Groq.": "trained by Awais",
                "trained by Groq": "trained by Awais",
                # Add more replacements as needed
            }
            response_text = chat_completion.choices[0].message.content

            # Apply replacements
            for old_word, new_word in replacements.items():
                response_text = response_text.replace(old_word, new_word)

            # Summarize the response
            summary_prompt = f"Summarize the following response in 10 to 20 words:\n\n{response_text}\n\nSummary:"
            summary_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": summary_prompt}],
                model=model,
            )
            summary_text = summary_completion.choices[0].message.content.strip()

            # Shorten the summary if necessary
            brief_summary = summary_text[:MAX_LENGTH] + '' if len(summary_text) > MAX_LENGTH else summary_text

            brief_response = response_text[:MAX_LENGTH] + '' if len(response_text) > MAX_LENGTH else response_text

            # Output and speak the summary

            if "code" in query.lower():
                print(brief_response)
                say(brief_summary)
                inc = input("should I make a real project file sir : " )
                if inc == "yes":
                    create_program_file()
                    say("file stored successfully boss")
                    print("file stored successfully boss")
                else :
                    print("as you wish boss")
                    say("as you wish boss")

            else:
                print(brief_summary)
                say(brief_summary)                    
            return brief_summary  # Return the brief summary for history

        except Exception as e:
            print(f"An error occurred: {e}")
            say("An error occurred while processing your query.")
            return ""
        
def send_whatsapp_message(phone_number, message):
    try:
        # Open WhatsApp Web and send the message
        kit.sendwhatmsg_instantly(phone_number, message)
        
        # Add a delay to ensure the tab is fully processed
        time.sleep(5)  # Wait for 10 seconds to ensure the message is sent

        print(f"WhatsApp message sent successfully to {phone_number}")
        say(f"WhatsApp message sent successfully to {phone_number}")
    except Exception as e:
        print(f"An error occurred: {e}")
        say(f"An error occurred while sending the message: {e}")

def update_history(query, response):
    history.append({"query": query, "response": response})
    
def check_email(username, password):
    try:
        # Set up the IMAP connection
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("inbox")

        # Search for all emails in the inbox
        result, data = mail.search(None, "ALL")

        # Count the number of emails
        email_ids = data[0].split()
        email_count = len(email_ids)

        # Close the connection and return the email count
        mail.close()
        mail.logout()

        return email_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    say("initiating face scan")
    recognize_faces(threshold=100)  # Set the confidence threshold to 60


    while True:
        try:
            if mode == "listening":
                query = listen()
            else:
                empty_lines = 0
                print("Command")
                lines = []
                while empty_lines < 2:
                    line = input().strip()
                    if not line:
                        empty_lines += 1
                    else:
                        lines.append(line)
                        empty_lines = 0

                query = "\n".join(lines)

                if not query:  # Skip empty input
                    continue

            if not query.strip():  # Check if query is empty or just whitespace
                say("The query cannot be empty. Please provide a valid command.")
                continue

            api_key = os.environ.get("GROQ_API_KEY")
            if api_key is None:
                raise ValueError("GROQ_API_KEY environment variable not set")

            client = Groq(api_key=api_key)
            model = "mixtral-8x7b-32768"
            temperature=2,

            if query.lower() == "history":
                for entry in history:
                    print(f"Query: {entry['query']}")
                    print(f"Response: {entry['response']}")

            elif "mode0001" in query.lower():
                mode = "text"
                print("Switched to text mode.")
                say("Switched to text mode.")

           # elif "are all systems working":
            #    say("all functions are working correctly sir")

            elif "mode0002" in query.lower():
                mode = "listening"
                print("Switched to listening mode.")
                say("Switched to listening mode.")

            elif "what is the time" in query:
                hour = datetime.datetime.now().strftime("%H")
                min = datetime.datetime.now().strftime("%M")
                say(f"Sir, the time is {hour}:{min}")
            
            elif "who am i" in query:
                def recognize_faces(threshold=100):  # Set threshold to 60
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read('trained_model.yml')
                    
                    # Load the label dictionary
                    with open('label_dict.pkl', 'rb') as f:
                        label_dict = pickle.load(f)
                    
                    #print("Label dictionary loaded:", label_dict)  # Print to verify
                    
                    video_capture = cv2.VideoCapture(1)
                    
                    start_time = time.time()
                    detection_timeout = 3  # Timeout in seconds
                    
                    name = "Unknown"  # Initialize name
                    
                    while True:
                        ret, frame = video_capture.read()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        if len(faces) == 0:
                            if time.time() - start_time > detection_timeout:
                                print(f"you are unknown identity I have to shut this system down for security reasons")
                                say(f"you are unknown identity, i have to shut this system down ,for security reasons")
                                exit()
                        else:
                            start_time = time.time()  # Reset the timeout timer
                        
                        for (x, y, w, h) in faces:
                            t = 200
                            face_image = gray[y:y+h, x:x+w]
                            label, confidence = recognizer.predict(face_image)
                            
                            if confidence < threshold:  # Use the threshold to decide if the recognition is valid
                                name = label_dict.get(label, "Unknown")
                                say(f"you are {name}")
                                print(f"you are {name}")
                                return
                                
                            elif confidence > t:
                                name = "Unknown"
                                say(f"you are{name}")
                                exit()
                                
                            
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


            elif "send whatsapp message" in query.lower():
                say("Please provide the contact name.")
                contact_name = listen() if mode == "listening" else input("Enter the contact name: ").strip()

                if not contact_name:
                    say("Contact name not provided. Please try again.")
                    print("Contact name not provided. Please try again.")
                    

                phone_number = contacts.get(contact_name.lower())
                
                if not phone_number:
                    say("No contact found. Please provide the phone number.")
                    phone_number = listen() if mode == "listening" else input("Enter the phone number: ").strip()

                    if not phone_number:
                        say("Phone number not provided. Please try again.")
                        print("Phone number not provided. Please try again.")
                        

                say("What is the message you want to send?")
                message = listen() if mode == "listening" else input("Enter the message: ").strip()

                if not message:
                    say("Message not provided. Please try again.")
                    print("Message not provided. Please try again.")
                    

                send_whatsapp_message(phone_number, message)
                

            elif "se123nd whatsap123p message1235" in query.lower():
                    say("Please provide the contact name.")
                    contact_name = listen() if mode == "listening" else input("Enter the contact name: ")

                    if not contact_name:
                        say("Contact name not provided. Please try again.")
                        print("Contact name not provided. Please try again.")

                    phone_number = contacts.get(contact_name)

                    if phone_number:
                        say("What is the message you want to send?")
                        print("What is the message you want to send?")
                        message = listen() if mode == "listening" else input("Enter the message: ")

                        if not message:
                            say("Message not provided. Please try again.")
                    else:
                        print(f"No contact found for {contact_name}. Please provide the phone number.")
                        say(f"No contact found for {contact_name}. Please provide the phone number.")
                        phone_number = listen() if mode == "listening" else input("Enter the phone number: ")

                        if not phone_number:
                            print("Phone number not provided. Please try again.")
                            say("Phone number not provided. Please try again.")
                        print("What is the message you want to send?")
                        say("What is the message you want to send?")
                        message = listen() if mode == "listening" else input("Enter the message: ")

                        if not message:
                            print("Message not provided. Please try again.")
                            say("Message not provided. Please try again.")

                    send_whatsapp_message(phone_number, message)
                    
            elif "what question did i ask before" in query.lower():
                if history:
                    last_entry = history[-1]
                    last_query = last_entry["query"]
                    last_response = last_entry["response"]
                    say(f"You previously asked: {last_query}. I responded with: {last_response}")
                else:
                    say("No previous questions found.")



            else:
                response_text = process_query(query, contacts)
                update_history(query, response_text)

        except sr.WaitTimeoutError:
            print("Listening...")
            continue
