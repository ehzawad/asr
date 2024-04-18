import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
import numpy as np

model_path = "/home/whisper/whisper-base-bn"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

data_directory = "/home/whisper/data"
max_duration = 600  # Maximum duration in seconds (e.g., 10 minutes)
target_sampling_rate = 16000  # Sampling rate expected by the Whisper model

# Iterate over the audio files in the data directory
for filename in os.listdir(data_directory):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        audio_path = os.path.join(data_directory, filename)
        audio, sr = sf.read(audio_path)

        # Check the audio duration
        duration = len(audio) / sr
        if duration > max_duration:
            print(f"Skipping {filename}: Audio duration exceeds the maximum limit.")
            continue

        # Convert audio data to float32
        audio = audio.astype(np.float32)

        # Resample the audio to the target sampling rate
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sampling_rate)

        inputs = processor(audio, sampling_rate=target_sampling_rate, return_tensors="pt")

        try:
            generated_ids = model.generate(inputs["input_features"], num_beams=4, max_length=256)
            transcription = processor.decode(generated_ids[0], skip_special_tokens=True)

            print(f"Transcription for {filename}:")
            print(transcription)
            print("---")
        except Exception as e:
            print(f"Error during inference for {filename}: {str(e)}")
            continue
