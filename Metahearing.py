import torch
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define the audio sampling rate
SAMPLING_RATE = 16000  # The model expects 16kHz audio input

# Function to capture audio from the microphone and return the audio signal
def record_audio(duration=5):
    print("Recording for {} seconds...".format(duration))
    audio = sd.rec(int(duration * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype=np.float32)
    sd.wait()  # Wait until the recording is finished
    return audio.flatten()

# Function to transcribe the recorded audio using the pre-trained model
def transcribe_audio(audio_input):
    # Process the audio input for the model
    inputs = processor(audio_input, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    
    # Perform inference with the model
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get the predicted token ids
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the token ids to text
    transcription = processor.decode(predicted_ids[0])
    
    return transcription

# Main loop to continuously capture and transcribe audio
def main():
    full_transcription = ""
    while True:
        # Capture audio from the microphone (for 2 seconds)
        audio_input = record_audio(duration=5)

        # Transcribe the audio
        transcription = transcribe_audio(audio_input)

        # Append the transcription to the full transcription
        full_transcription += transcription + " "

        # Display the ongoing transcription in the terminal
        print("Current Transcription:", full_transcription.strip())

        # Optionally, you can add a break condition if needed
        if "exit" in transcription.lower():
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
