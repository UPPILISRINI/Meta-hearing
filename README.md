Meta Hearing

Meta Hearing is an intelligent speech-recognition system designed to process real-time audio, convert speech to text, classify voice commands, and integrate seamlessly with microcontroller-based IoT devices such as ESP32. The project focuses on lightweight models, fast inference, and practical deployment in embedded systems and edge devices.

🚀 Features

🎤 Real-Time Speech Recognition

🔊 Custom Command Detection (Keyword/Hotword)

⚡ Optimized for ESP Boards (ESP32/ESP8266)

🧠 Lightweight ML Models for Edge Deployment

🔧 Modular Architecture (Training → Inference → Deployment)

🎛️ Configurable Voice Commands & Thresholds

📡 Serial/Wi-Fi Communication with ESP

🗂️ Includes Python + Firmware Codebase

📁 Project Structure

meta-hearing/
│


├── src/

│   ├── model/            # Pretrained or custom models

│   ├── audio/            # Audio processing & feature extraction

│   ├── esp/              # ESP32/ESP8266 firmware

│   ├── utils/            # Helper scripts & tools

│   └── main.py           # Desktop/PC-based speech recognition

│

├── dataset/              # Training data (optional)

│

├── docs/

│   └── setup_guide.md    # Hardware + software setup

│

├── requirements.txt      # Python dependencies

├── LICENSE

└── README.md

🛠️ Tech Stack
Core

Python

TensorFlow / PyTorch (depending on model)

NumPy, Librosa

ESP-IDF / Arduino Framework

Optional

Streamlit (UI Demo)

Flask/FastAPI (API endpoint)

🔧 Installation

1️⃣ Clone the repository
git clone https://github.com/<your-username>/meta-hearing.git
cd meta-hearing

2️⃣ Install Python dependencies
pip install -r requirements.txt

3️⃣ Run the speech recognition app
python src/main.py

📡 ESP Setup (Firmware)

Open Arduino IDE or ESP-IDF

Install ESP32 board packages

Connect ESP board

Open the firmware located in:

src/esp/


Configure:

Wi-Fi credentials (if used)

GPIO pins for microphone modules

Baud rate

Upload the code to your ESP board.

🧠 How Meta Hearing Works

Capture microphone input (PC or ESP)

Preprocess audio → noise removal + mel spectrogram

Model predicts:

Speech-to-text (transcription)

or Command classification

Output is sent back to PC or IoT devices

ESP executes actions based on recognized voice commands

🏗️ Training Your Own Model

To train a custom command model:

python src/model/train.py


Your dataset should follow:


dataset/

 ├── command_1/
 
 ├── command_2/
 
 ├── command_3/
 
 └── ...


Export the trained model to the src/model/ folder.

📊 Roadmap

 Add offline noise suppression

 Improve accuracy for low-power devices

 Add support for wake-word detection

 Add Web UI for testing

 Model quantization for ESP32


📜 License

This project is licensed under MIT License.
You are free to modify and use it in personal or commercial projects.

🙌 Acknowledgements

Google Speech Commands Dataset

TensorFlow & PyTorch audio processing tutorials

ESP-IDF Documentation
