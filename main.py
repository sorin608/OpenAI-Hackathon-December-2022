import os

import sounddevice as sd
from scipy.io.wavfile import write

import torch
import whisper
import numpy as np 

import openai
from dotenv import load_dotenv
load_dotenv()

import pyperclip


def record(duration):
        fs = 44100
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        print("Recording audio - Speak now!")
        sd.wait()
        print("Recording complete")
        write('output.mp3', fs, myrecording)

def transcribe():
        torch.cuda.is_available()
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        model = whisper.load_model("small", device = DEVICE)

        audio = whisper.load_audio("output.mp3")

        audio = whisper.pad_or_trim(audio) 

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 =False)

        result = whisper.decode(model, mel, options)

        print(result.text)

        result = model.transcribe('output.mp3')

        print(result["text"])
        return result["text"]
