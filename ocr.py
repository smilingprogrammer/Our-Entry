import cv2
import time
import requests
import google.generativeai as genai
import pyttsx3

# Constants for APIs
OCR_API_URL = "https://www.imagetotext.info/api/imageToText"
OCR_API_KEY = ""
GEMINI_API_KEY = ""

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Setup Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    print("Gemini:", text)
    engine.say(text)
    engine.runAndWait()

def clean_text_with_gemini(raw_text):
    prompt = f"You are a helpful assistant for a blind user.I have given u a text extracted by ocr extract the most useful and meaningful text in short and concise manner.Here is the extracted text:\n\n{raw_text}"
    response = model.generate_content([prompt])
    return response.text

def capture_image(path="captured_image.jpg"):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return None

    print("Starting camera... Live preview for 3 seconds.")
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam Preview", frame)

        if time.time() - start_time > 3:
            cv2.imwrite(path, frame)
            print(f"Image captured and saved as '{path}'")
            break

        if cv2.waitKey(1) == 27:  # ESC to exit early
            print("Early exit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return path

def call_ocr_api(image_path):
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        headers = {
            "Authorization": f"Bearer {OCR_API_KEY}"
        }
        print("Sending image to OCR API...")
        response = requests.post(OCR_API_URL, files=files, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if not data.get("error"):
            return data.get("result", "")
        else:
            print(f"OCR API Error: {data.get('message')}")
    else:
        print(f"HTTP Error from OCR API: {response.status_code}")
        print(response.text)
    return None

def main():
    img_path = capture_image()
    if img_path is None:
        return

    raw_text = call_ocr_api(img_path)
    if raw_text:
        print("Raw OCR Text:")
        print(raw_text)

        cleaned_text = clean_text_with_gemini(raw_text)
        speak(cleaned_text)
    else:
        speak("Sorry, I could not extract any text from the image.")

if __name__ == "__main__":
    main()
