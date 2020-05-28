from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext, loader, Context
from django.contrib.auth.decorators import login_required

from .base_camera import Camera as Localcamera
# from .base_camera_opencv import Camera as Opencvcamera

from gpiozero import Robot, DistanceSensor, Servo, LED
from gpiozero import DigitalInputDevice
from signal import pause
import os

import time
import threading
import cv2
import numpy as np


try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident



# sensor declare
# chaoshengbo = DistanceSensor(27, 22)
robot = Robot(left=(5, 6), right=(23, 24))
servo = Servo(26)
ir1 = DigitalInputDevice(25)
ir2 = DigitalInputDevice(16)


# script workspace
scriptfolder = os.path.dirname(os.path.dirname(__file__)) + '/script/'




class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()



class BaseCamera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    event = CameraEvent()

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # start background frame thread
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame."""
        BaseCamera.last_access = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event.wait()
        BaseCamera.event.clear()

        return BaseCamera.frame

    @staticmethod
    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses.')

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            BaseCamera.frame = frame
            BaseCamera.event.set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any clients asking for frames in
            # the last 10 seconds then stop the thread
            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        BaseCamera.thread = None


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
#        face_cascade = cv2.CascadeClassifier('/home/pi/work/django/dragoncar/dragoncar/haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier('/home/pi/work/django/dragoncar/dragoncar/q-320-3-15.xml')
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        framex = camera.get(3)
        framey = camera.get(4)
        centerx = framex/2
        centery = framey/2

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            # read current frame
            _, img = camera.read()

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)

            moveinfor="move:"
            reccenw="w:"
            reccenh="h:"

            for (x,y,w,h) in faces:

                if (len(faces) > 1):
                    break
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                face_area = img[y:y+h, x:x+w]
                reccenx = x + w/2
                recceny = y + h/2
                pianlix = reccenx - centerx
                pianliy = recceny - centery
                reccenw = str(w)
                reccenh = str(h)
                if (pianlix > 0):
                  inforx = " zuo " + str(abs(pianlix))
#          print(inforx)
                  robot.left()
                  time.sleep(0.2)
                  robot.stop()
                else:
                  inforx = " you " + str(abs(pianlix))
#          print(inforx)
                  robot.left()
                  time.sleep(0.2)
                  robot.stop()

                bilv = round(w/framex,2)
                if (bilv > 0.5):
                  robot.stop()
                else:
                  robot.forward(speed=0.75*(1-bilv))

            printtext = str(len(faces)) + " people " + str(framex) + " x " + str(framey) + " " + reccenw + " " + reccenh
            cv2.putText(img, printtext, (10,30), font, 0.7,(255,255,255),2,cv2.LINE_AA)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()




def stopstatus():
  os.system("ps -aux | grep dragoncar | cut -d ' ' -f9 | xargs kill -9")

# Create your views here.

def fixeddistance():
    os.system("python3 " + scriptfolder + "dragoncar-fixeddistance.py")


def autocar(request):
    if "carfixeddistance" in request.POST:
        print("fixed distance")
        fixeddistance()
    if "carstop" in request.POST:
        stopstatus()
#        robot.stop()

    return render(request, 'dragoncar/autocar.html')




def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(Localcamera()),
        content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
  return render(request, 'index.html')


def wificontrol(request):
  if (request.POST.get('power') != None):
    carpower = float(request.POST.get('power'))/1000
    print(carpower)

    if "stop" in request.POST:
      robot.stop()

    if "up" in request.POST:
      robot.forward(speed=carpower)
    if "down" in request.POST:
      robot.backward(speed=carpower)
    if "left" in request.POST:
      robot.left(speed=carpower)
      time.sleep(0.3)
      robot.stop()
      time.sleep(0.1)
      robot.forward(speed=carpower)
    if "right" in request.POST:
      robot.right(speed=carpower)
      time.sleep(0.3)
      robot.stop()
      time.sleep(0.1)
      robot.forward(speed=carpower)

    if "littleup" in request.POST:
      robot.forward(speed=carpower)
      time.sleep(0.1)
      robot.stop()
    if "littledown" in request.POST:
      robot.backward(speed=carpower)
      time.sleep(0.1)
      robot.stop()
    if "littleleft" in request.POST:
      robot.left(speed=carpower)
      time.sleep(0.1)
      robot.stop()
    if "littleright" in request.POST:
      robot.right(speed=carpower)
      time.sleep(0.1)
      robot.stop()

  return render(request, 'dragoncar/wificontrol.html')  


def videocar(request):
  if (request.POST.get('power') != None):
    carpower = float(request.POST.get('power'))/1000
    print(carpower)

    if "stop" in request.POST:
      robot.stop()

    if "up" in request.POST:
      robot.forward(speed=carpower)
    if "down" in request.POST:
      robot.backward(speed=carpower)
    if "left" in request.POST:
      robot.left(speed=carpower)
      time.sleep(0.3)
      robot.stop()
      time.sleep(0.1)
      robot.forward(speed=carpower)
    if "right" in request.POST:
      robot.right(speed=carpower)
      time.sleep(0.3)
      robot.stop()
      time.sleep(0.1)
      robot.forward(speed=carpower)

    if "littleup" in request.POST:
      robot.forward(speed=carpower)
      time.sleep(0.1)
      robot.stop()
    if "littledown" in request.POST:
      robot.backward(speed=carpower)
      time.sleep(0.1)
      robot.stop()
    if "littleleft" in request.POST:
      robot.left(speed=carpower)
      time.sleep(0.1)
      robot.stop()
    if "littleright" in request.POST:
      robot.right(speed=carpower)
      time.sleep(0.1)
      robot.stop()

  return render(request, 'dragoncar/videocar.html')


