import cv2
import numpy as np
import pytesseract
import speech_recognition as sr
import pyttsx3
import datetime
import torch
print(torch.__version__)
import requests
from googletrans import Translator
from geopy.geocoders import Nominatim
from apikey import api_data
import pytesseract
import threading
import tkinter as tk
from PIL import Image, ImageTk
import base64

class CameraApp:
    def __init__(self, window, camera_ip):
        self.window = window
        self.window.title("Vision AI Camera Feed")

        self.camera_ip = camera_ip
        self.cap = cv2.VideoCapture(camera_ip)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            speak("Could not open the camera. Please check the IP address and connection.")
            print(f"Failed to open camera at: {camera_ip}")
            self.window.destroy()
            return

        self.label = tk.Label(window)
        self.label.pack()

        quit_btn = tk.Button(window, text="Quit (Close Camera)", command=self.close)
        quit_btn.pack(pady=10)

        # Add these three lines to bind keys for exiting
        window.bind('<q>', lambda event: self.close())
        window.bind('<Q>', lambda event: self.close())
        window.bind('<Escape>', lambda event: self.close())

        # Force window focus so it receives keyboard events immediately
        window.focus_force()

        self.running = True
        threading.Thread(target=self.video_loop, daemon=True).start()

        self.window.protocol("WM_DELETE_WINDOW", self.close)


    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            detect_objects(frame)  # Your object detection function modifies frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.window.update_idletasks()
            self.window.update()

    def close(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()
        speak("Camera closed.")

def run_camera_app():
    root = tk.Tk()
    camera_ip = 'http://192.168.130.240:8080/video'  # Replace with your IP
    app = CameraApp(root, camera_ip)
    
    root.mainloop()
# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\pooja\OneDrive\Desktop\Leo_AI\tesseract.exe'  # Update this path if necessary 

# Set up API key for Gemini AI
GENAI_API_KEY = api_data
gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Initialize Speech Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty("rate", 180)

def speak(audio):
    print(f"Speaking: {audio}")
    engine.say(audio)
    engine.runAndWait()

def command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=7)
        except sr.WaitTimeoutError:
            print("Listening timed out.")
            return "none"

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User  said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand. Please repeat.")
        return "none"
    except sr.RequestError as e:
        speak("There was a problem with the speech recognition service.")
        return "none"
    except Exception as e:
        speak(f"An error occurred: {e}")
        return "none"

def get_time():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")

def get_date():
    today = datetime.datetime.now()
    return today.strftime("%B %d, %Y")

# Update the paths according to your directory structure
weights_path = "e:\\Leo_AI\\yolov3.weights"
config_path = "e:\\Leo_AI\\yolov3.cfg"
coco_names_path = "e:\\Leo_AI\\coco.names"

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class labels
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

# Check if the output layers are scalar or array and handle accordingly
if isinstance(unconnected_out_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_out_layers - 1]]

detected_objects_previous = set()  # To track previously detected objects

detected_objects_previous = set()  # To track previously detected objects

def detect_objects(frame):
    global detected_objects_previous  # Use the global variable to track previous detections
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = set()  # To track objects detected in this frame

    # Check if indexes is empty
    if indexes is not None and len(indexes) > 0:
        if isinstance(indexes, tuple):
            indexes = indexes[0]  # Access the first element if it's a tuple

        for i in indexes.flatten():  # Now we can safely flatten
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            detected_objects.add(label)  # Add detected label to the set

            # Speak out the detected object if it's new
            if label not in detected_objects_previous:
                speak(f"I see a {label}.")
                detected_objects_previous.add(label)  # Add to previous detections

    # Update the previous detections to the current frame's detections
    detected_objects_previous = detected_objects_previous.intersection(detected_objects)

    return detected_objects

# def open_camera():
#     camera_ip = 'http://192.168.1.12:8080/video'
#     cap = cv2.VideoCapture(camera_ip)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#     if not cap.isOpened():
#         speak("Could not open the camera. Please check the IP address and connection.")
#         print("Failed to open camera at:", camera_ip)
#         return

#     speak("Camera is now open. Press 'q' or ESC to close the camera.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             speak("Failed to grab frame.")
#             print("Failed to grab frame from camera.")
#             break

#         detect_objects(frame)
#         cv2.imshow("Camera", frame)

#         key = cv2.waitKey(30) & 0xFF
#         if key == ord('q') or key == 27:  # 27 is ESC key
#             print("Exit key pressed, closing camera.")
#             break

#         # Exit if window closed manually
#         if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
#             print("Window closed manually, exiting.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()


def read_text_with_gemini(frame):
    """Use Gemini API to detect and read text from an image"""
    try:
        # Convert the frame to base64 encoding (required for API)
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare the request to Gemini API
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GENAI_API_KEY
        }
        
        # Build the request body with the image
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": "Please extract and read any text visible in this image."},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        # Send the request to Gemini API
        response = requests.post(gemini_url, headers=headers, json=data)
        response.raise_for_status()
        
        # Extract the text from response
        result = response.json()
        
        # The exact structure may need to be adjusted based on the actual API response
        extracted_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        
        return extracted_text if extracted_text else "No text detected."
        
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        return "Error detecting text with Gemini."
def translate_text(text, dest_lang='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=dest_lang).text
    speak("Translated text: " + translated_text)
    return translated_text
def read_text():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        speak("Could not open the camera.")
        return ""

    ret, frame = cap.read()  # Capture a frame
    cap.release()  # Release the camera

    if not ret:
        speak("Failed to grab frame for text reading.")
        return ""

    # Use Gemini instead of pytesseract
    text = read_text_with_gemini(frame)
    
    if text and text != "No text detected." and text != "Error detecting text with Gemini.":
        speak("The text says: " + text)
    else:
        speak(text)  # Will say "No text detected" or error message
        
    return text
def currency_det():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        speak("Could not open the camera.")
        return ""

    ret, frame = cap.read()  # Capture a frame
    cap.release()  # Release the camera

    if not ret:
        speak("Failed to grab frame for currency detecting.")
        return ""

    # Use Gemini instead of pytesseract
    text = read_text_with_gemini(frame)
    
    if text and text != "No currency detected." and text != "Error detecting currency with Gemini.":
        speak("The currency is: " + text)
    else:
        speak(text)  # Will say "No text detected" or error message
        
    return text


def get_weather():
    api_key = "e9f54be0915abb6f5a99218de12613ce"  # Make sure to keep this secure
    city = "Delhi"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        weather = response.json()['weather'][0]['description']
        temp = response.json()['main']['temp']
        speak(f"The current weather in {city} is {weather} with a temperature of {temp} degrees Celsius.")
    except requests.exceptions.RequestException as e:
        speak(f"Could not retrieve weather data: {e}")

def get_location():
    try:
        geolocator = Nominatim(user_agent="myGeocoder")
        location = geolocator.geocode("Delhi")
        return location.address if location else "Location not found."
    except Exception as e:
        speak(f"An error occurred while retrieving location: {e}")
        return "Location not found."

def get_ip_info():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        ip_address = data.get("ip", "IP not found")
        location = data.get("loc", "Location not found")  # This returns latitude and longitude
        city = data.get("city", "City not found")
        region = data.get("region", "Region not found")
        country = data.get("country", "Country not found")
        return ip_address, f"{city}, {region}, {country}"
    except Exception as e:
        print(f"Error retrieving IP information: {e}")
        return "IP not found", "Location not found"

def emergency_alert():
    # Get the user's IP address and location
    ip_address, location = get_ip_info()
    
    # Speak the emergency alert with location and IP address
    speak(f"Emergency alert sent with location: {location} and IP address: {ip_address}")

def main():
    speak("Hello! I am Vision AI. How can I assist you?")
    while True:
        query = command()
        if query == "none":
            continue

        if "exit" in query or "stop" in query:
            speak("Goodbye!")
            break
        elif "time" in query:
            current_time = get_time()
            speak(f"The current time is {current_time}.")
        elif "date" in query:
            current_date = get_date()
            speak(f"Today's date is {current_date}.")
#         elif "detect objects" in query:
#             open_camera()
        elif "read text" in query:
            text = read_text()
            speak(text)
        elif "detect currency" in query:
            text = currency_det()
            speak(text)
        elif "translate" in query:
            text = command()
            translated_text = translate_text(text, 'fr')
        elif "weather" in query:
            get_weather()
        elif "emergency" in query:
            emergency_alert()
        elif "open camera" in query or "detect objects" in query:
            run_camera_app()
        else:
            speak("Command not recognized.")

if __name__ == "__main__":
    main()