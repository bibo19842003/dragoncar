from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext, loader, Context
from django.contrib.auth.decorators import login_required
import urllib.request

# vosk
from vosk import Model, KaldiRecognizer
import sys
import json
import os
import wave
import configparser

# common
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# vosk
vosk_model = Model(BASE_DIR + "/model/vosk")

# parse config file
dragon_cf = configparser.ConfigParser()
dragon_cf.read(BASE_DIR + '/dragonconfig/voice_rec.ini')


def home(request):
  return render(request, 'dragonhome/home.html')


def homesensor(request):
  return render(request, 'dragonhome/homesensor.html')


def upload_home(request):
    if request.method == "POST":
        myFile =request.FILES.get("myfile", None)
        if not myFile:
            print("no files for upload!")
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("media/voice",myFile.name),'wb+')
        for chunk in myFile.chunks():
            destination.write(chunk)
        destination.close()

        rec = KaldiRecognizer(vosk_model, 16000)
        wf = wave.open(BASE_DIR + '/media/voice/voicehome.wav', "rb")

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                rec.Result()

        data = json.loads(rec.FinalResult())
        voicetext = data['text']

        print(voicetext)

        led_red_open = dragon_cf['voicerec']['led_red_open'].split(',')
        led_red_close = dragon_cf['voicerec']['led_red_close'].split(',')
        led_yellow_open = dragon_cf['voicerec']['led_yellow_open'].split(',')
        led_yellow_close = dragon_cf['voicerec']['led_yellow_close'].split(',')
        led_green_open = dragon_cf['voicerec']['led_green_open'].split(',')
        led_green_close = dragon_cf['voicerec']['led_green_close'].split(',')
        led_all_open = dragon_cf['voicerec']['led_all_open'].split(',')
        led_all_close = dragon_cf['voicerec']['led_all_close'].split(',')

        if voicetext in led_red_open:
            with urllib.request.urlopen('http://192.168.1.9/?rledstatus=1') as response:
                html = response.read()
        elif voicetext in led_red_close:
            with urllib.request.urlopen('http://192.168.1.9/?rledstatus=0') as response:
                html = response.read()
        elif voicetext in led_yellow_open:
            with urllib.request.urlopen('http://192.168.1.9/?yledstatus=1') as response:
                html = response.read()

        elif voicetext in led_yellow_close:
            with urllib.request.urlopen('http://192.168.1.9/?yledstatus=0') as response:
                html = response.read()
        elif voicetext in led_green_open:
            with urllib.request.urlopen('http://192.168.1.9/?gledstatus=1') as response:
                html = response.read()
        elif voicetext in led_green_close:
            with urllib.request.urlopen('http://192.168.1.9/?gledstatus=0') as response:
                html = response.read()
        elif voicetext in led_all_open:
            with urllib.request.urlopen('http://192.168.1.9/?gledstatus=1&yledstatus=1&rledstatus=1') as response:
                html = response.read() 
        elif voicetext in led_all_close:
            with urllib.request.urlopen('http://192.168.1.9/?gledstatus=0&yledstatus=0&rledstatus=0') as response:
                html = response.read() 

        return HttpResponse("upload over!")


