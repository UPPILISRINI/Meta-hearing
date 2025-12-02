import speech_recognition as sr
import pyrebase

# Firebase Configuration (Replace with your actual Firebase credentials)
firebase_config = {
    "apiKey": "AIzaSyCXeTza_jl9A95Thgb4EVbzcdxSMMIk8wM",
    "authDomain": "project-259c7.firebaseapp.com",
    "databaseURL": "https://project-259c7-default-rtdb.firebaseio.com",
    "storageBucket": "project-259c7.firebasestorage.app"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()
full_text = ""

def speech_to_text():
    global full_text
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Listening continuously... (Press Ctrl+C to stop)")

    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Speak now...")
                audio = recognizer.listen(source)

            # Convert Speech to Text
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")

            # Append new text
            full_text += " " + text

            # Send updated text to Firebase
            db.child("oled").set({"string": full_text.strip()})
            print("Updated text sent to Firebase!")

        except sr.UnknownValueError:
            print("Could not understand the audio, try again.")
        except sr.RequestError:
            print("Could not connect to the speech recognition service.")
        except KeyboardInterrupt:
            print("\nStopping speech recognition...")
            break  # Stop when user presses Ctrl+C

if __name__ == "__main__":
    speech_to_text()
