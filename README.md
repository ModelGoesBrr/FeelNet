# FeelNet
Understanding Emotions through Audio-Visual AI
I'll create a comprehensive multimodal emotion recognition system that combines audio and visual cues for emotion detection.I've created a comprehensive multimodal emotion recognition system! Here's what it includes:

## Key Components

**1. Facial Emotion Recognition (CNN)**
- 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- 3-layer CNN architecture with batch normalization and dropout
- Uses OpenCV Haar Cascades for face detection
- Processes 48x48 grayscale face images

**2. Audio Emotion Recognition (LSTM)**
- 5 emotions: Angry, Happy, Sad, Neutral, Fear
- LSTM-based architecture for sequential audio feature processing
- Extracts features using Librosa: MFCCs, chroma, mel spectrogram, spectral contrast
- Processes 3-second audio segments

**3. Multimodal Fusion**
- Weighted averaging (60% visual, 40% audio by default)
- Combines predictions from both modalities
- Provides unified emotion classification

## Features

- **Real-time processing** from webcam and microphone
- **Face detection** using OpenCV
- **Audio feature extraction** with Librosa
- **Deep learning models** (CNN + LSTM)
- **Prediction fusion** for improved accuracy
- Visual feedback with bounding boxes and confidence scores

## To Use in Production

1. **Install dependencies**:
```bash
pip install opencv-python numpy librosa tensorflow sounddevice scipy
```

2. **Train the models** on datasets:
   - **Visual**: FER2013, AffectNet, or CK+
   - **Audio**: RAVDESS, IEMOCAP, or TESS

3. **Load pre-trained weights**:
```python
mer.facial_model.model.load_weights('facial_model.h5')
mer.audio_model.model.load_weights('audio_model.h5')
```

The models are currently untrained and would need training on appropriate datasets for accurate emotion recognition!
## "Prototype / Proof of Concept"

- **IT IS STILL UNTRAINED.
