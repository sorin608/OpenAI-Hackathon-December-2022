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

def generate_email(text):
        openai.api_key=os.getenv("OPENAI_API_KEY")
        response=openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Create a kind and formal email for this reason: {text}",
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
        )
        return response["choices"][0]["text"]

def main():
        record(8)
        text=transcribe()
        email=generate_email(text)
        print(email)
        pyperclip.copy(email)

if __name__=="__main__":
    main()