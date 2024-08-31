import os
import requests

def generate_image_via_api(prompt, filename):
    # URL for the stabilityai/stable-fast-3d model
    API_URL = "https://api-inference.huggingface.co/models/goofyai/3d_render_style_xl"
    headers = {
        "Authorization": "Bearer your-api-key"
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

while True:
    prompt = input("Enter your prompt : ")
    if prompt.lower() == 'exit':
        break
    imgname = input("Enter the name of the image file (e.g., 'my_image'): ")
    filename = imgname + ".png"  # Combine the input name with ".png" extension
    generate_image_via_api(prompt, filename)
