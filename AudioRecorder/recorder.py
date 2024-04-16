import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

def record_audio(duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished")
    return recording

def save_wav(file_name, data, fs):
    print(f"Saving recording to {file_name}")
    wav.write(file_name, fs, data)
    print("File saved successfully")

# Parameters
fs = 44100  # Sample rate
duration = 5  # seconds
file_name = 'output.wav'

# Record audio
audio = record_audio(duration, fs)

# Save the recorded audio to a WAV file
save_wav(file_name, audio, fs)
