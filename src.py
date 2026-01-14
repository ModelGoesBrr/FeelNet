import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import sounddevice as sd
import queue
import threading
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

class FacialEmotionModel:
    """CNN-based facial emotion recognition model"""
    
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = self._build_model()
        
    def _build_model(self):
        """Build CNN architecture for facial emotion recognition"""
        model = models.Sequential([
            # First conv block
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Third conv block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    
    def predict(self, face_img):
        """Predict emotion from face image"""
        preprocessed = self.preprocess_face(face_img)
        predictions = self.model.predict(preprocessed, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        return self.emotions[emotion_idx], confidence, predictions[0]


class AudioEmotionModel:
    """LSTM-based audio emotion recognition model"""
    
    def __init__(self):
        self.emotions = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fear']
        self.model = self._build_model()
        self.sample_rate = 22050
        
    def _build_model(self):
        """Build LSTM architecture for audio emotion recognition"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(None, 40)),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def extract_features(self, audio_data, sr):
        """Extract MFCC features from audio"""
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        
        # Extract additional features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        
        # Combine features
        features = np.vstack([mfccs, chroma, mel[:20], spectral_contrast])
        
        return features.T
    
    def predict(self, audio_data, sr):
        """Predict emotion from audio"""
        features = self.extract_features(audio_data, sr)
        features = np.expand_dims(features, axis=0)
        
        predictions = self.model.predict(features, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        return self.emotions[emotion_idx], confidence, predictions[0]


class MultimodalEmotionRecognition:
    """Combine visual and audio emotion recognition"""
    
    def __init__(self):
        self.facial_model = FacialEmotionModel()
        self.audio_model = AudioEmotionModel()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion mapping for fusion
        self.emotion_mapping = {
            'Angry': 'Angry',
            'Happy': 'Happy',
            'Sad': 'Sad',
            'Neutral': 'Neutral',
            'Fear': 'Fear',
            'Surprise': 'Surprise',
            'Disgust': 'Disgust'
        }
        
    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def fuse_predictions(self, visual_emotion, visual_conf, visual_probs,
                        audio_emotion, audio_conf, audio_probs, 
                        visual_weight=0.6, audio_weight=0.4):
        """Fuse visual and audio predictions with weighted averaging"""
        
        # Create unified emotion space
        all_emotions = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fear', 'Surprise', 'Disgust']
        fused_probs = np.zeros(len(all_emotions))
        
        # Map visual predictions
        for i, emotion in enumerate(self.facial_model.emotions):
            idx = all_emotions.index(emotion)
            fused_probs[idx] += visual_probs[i] * visual_weight
        
        # Map audio predictions
        for i, emotion in enumerate(self.audio_model.emotions):
            idx = all_emotions.index(emotion)
            fused_probs[idx] += audio_probs[i] * audio_weight
        
        # Normalize
        fused_probs = fused_probs / (visual_weight + audio_weight)
        
        final_emotion_idx = np.argmax(fused_probs)
        final_confidence = fused_probs[final_emotion_idx]
        
        return all_emotions[final_emotion_idx], final_confidence, fused_probs
    
    def process_video_frame(self, frame):
        """Process single video frame for emotion detection"""
        faces = self.detect_faces(frame)
        results = []
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion, confidence, probs = self.facial_model.predict(face_roi)
            
            results.append({
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probs
            })
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, results
    
    def run_realtime_demo(self, use_audio=True, duration=30):
        """Run real-time emotion recognition demo"""
        cap = cv2.VideoCapture(0)
        
        print("Starting Multimodal Emotion Recognition...")
        print("Press 'q' to quit")
        
        # Audio recording setup
        audio_queue = queue.Queue()
        audio_data = []
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())
        
        # Start audio stream if enabled
        if use_audio:
            stream = sd.InputStream(callback=audio_callback,
                                   channels=1,
                                   samplerate=22050)
            stream.start()
        
        frame_count = 0
        audio_buffer_size = 22050 * 3  # 3 seconds of audio
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process video
            processed_frame, visual_results = self.process_video_frame(frame)
            
            # Process audio periodically
            if use_audio and frame_count % 30 == 0:  # Every ~1 second
                while not audio_queue.empty():
                    audio_data.extend(audio_queue.get().flatten())
                
                if len(audio_data) >= audio_buffer_size:
                    audio_segment = np.array(audio_data[-audio_buffer_size:])
                    audio_emotion, audio_conf, audio_probs = \
                        self.audio_model.predict(audio_segment, 22050)
                    
                    # Display audio emotion
                    cv2.putText(processed_frame, f"Audio: {audio_emotion} ({audio_conf:.2f})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (255, 0, 0), 2)
                    
                    # Fuse predictions if face detected
                    if visual_results:
                        fused_emotion, fused_conf, _ = self.fuse_predictions(
                            visual_results[0]['emotion'],
                            visual_results[0]['confidence'],
                            visual_results[0]['probabilities'],
                            audio_emotion,
                            audio_conf,
                            audio_probs
                        )
                        
                        cv2.putText(processed_frame, f"Fused: {fused_emotion} ({fused_conf:.2f})",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (0, 0, 255), 2)
            
            cv2.imshow('Multimodal Emotion Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        if use_audio:
            stream.stop()
        cv2.destroyAllWindows()


# Demo usage
if __name__ == "__main__":
    # Initialize the multimodal system
    mer = MultimodalEmotionRecognition()
    
    print("=" * 60)
    print("MULTIMODAL EMOTION RECOGNITION SYSTEM")
    print("=" * 60)
    print("\nFeatures:")
    print("- Visual: CNN-based facial emotion detection (7 emotions)")
    print("- Audio: LSTM-based speech emotion analysis (5 emotions)")
    print("- Fusion: Weighted combination of both modalities")
    print("\nSupported Emotions:")
    print("Visual:", mer.facial_model.emotions)
    print("Audio:", mer.audio_model.emotions)
    print("\n" + "=" * 60)
    
    # Run demo
    try:
        mer.run_realtime_demo(use_audio=True)
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: This demo requires a webcam and microphone.")
        print("For production use, train the models on datasets like:")
        print("- FER2013 or AffectNet (facial emotions)")
        print("- RAVDESS or IEMOCAP (audio emotions)")