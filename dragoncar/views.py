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
import face_recognition


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

        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        framex = camera.get(3)
        framey = camera.get(4)
        centerx = framex/2
        centery = framey/2
        font = cv2.FONT_HERSHEY_SIMPLEX

# Load a sample picture and learn how to recognize it.
        print("load face image: begin begin")
        bibo_image = face_recognition.load_image_file("/home/pi/work/django/dragoncar/dragoncar/a13.jpg")
        print("load face image: end end")
        bibo_face_encoding = face_recognition.face_encodings(bibo_image)[0]

# Create arrays of known face encodings and their names
        known_face_encodings = [
            bibo_face_encoding
        ]
        known_face_names = [
            "bibo"
        ]

# Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            # read current frame
            _, frame = camera.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.50)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
#            for (top, right, bottom, left), name in zip(face_locations, face_names):
            for (x, y, w, h), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                x *= 4
                y *= 4
                w *= 4
                h *= 4
#                name = name + str(x) + " " + str(y) + " " + str(w) + " " + str(h) 
                # Draw a box around the face
                cv2.rectangle(frame, (h, x), (y, w), (0, 255, 255), 2)
                cv2.putText(frame, name, (h + 6, w - 6), font, 1.0, (255, 0, 255), 2)

            # dragoncar turn left or right and go
            if ((face_names.count(known_face_names[0]) == 1) and (name == known_face_names[0])):
                lr = (h+y)/2
                if (lr > centerx):
                  robot.left()
                  time.sleep(0.2)
                  robot.stop
                  print("left")
                if (lr < centerx):
                  robot.right()
                  time.sleep(0.2)
                  robot.stop()
                  print("right")

                bilv = round((y-h)/(w-x)*0.5, 2)
                if (bilv > 0.5):
                  robot.stop()
                  print("stop stop")
                else:
                  robot.forward(speed=0.75*(1-bilv))
                  print("go go go")
            else:
                robot.stop()
                print("no name , stop!!!")

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()

# face_names.count(known_face_names[0]) == 1


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


# opencv camera
def followme_feed(request):
    return StreamingHttpResponse(gen(Camera()),
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


def followme(request):
  return render(request, 'dragoncar/followme.html')


def voicecontrol(request):
  return render(request, 'dragoncar/voicecontrol.html')


def uploadfile(request):
  return render(request, 'dragoncar/uploadfile.html')


def upload_file(request):
    if request.method == "POST":
        myFile =request.FILES.get("myfile", None)
        if not myFile:
            print("no files for upload!")
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("media/voice",myFile.name),'wb+')
        for chunk in myFile.chunks():
            destination.write(chunk)
        destination.close()
        print("upload over!")
        return HttpResponse("upload over!")
