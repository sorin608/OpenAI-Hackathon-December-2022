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