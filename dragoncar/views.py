from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext, loader, Context
from django.contrib.auth.decorators import login_required

from .base_camera import Camera as Localcamera
from .base_camera import Camera as Detectcarnumbercamera
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
import sys
import tarfile
from hyperlpr import *
from PIL import Image, ImageDraw, ImageFont
import pyzbar.pyzbar as pyzbar

from .models import Uploadimage

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident

from threading import Thread
import importlib.util
from datetime import datetime

# vosk
from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import json

# common
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# vosk
vosk_model = Model(BASE_DIR + "/model/vosk")

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# sensor declare
robot = Robot(left=(5, 6), right=(23, 24))
servoud = Servo(26)
servolr = Servo(17)
ir1 = DigitalInputDevice(25)
ir2 = DigitalInputDevice(16)


# script workspace
scriptfolder = os.path.dirname(os.path.dirname(__file__)) + '/script/'


# servo position
# servo: up and down
servoudvalue = 0
servolrvalue = 0



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
                  robot.right(0.2)
                  time.sleep(0.1)
                  robot.stop()
                  print("left")
                if (lr < centerx):
                  robot.left(0.2)
                  time.sleep(0.1)
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


class Recface(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Recface.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Recface, self).__init__()

    @staticmethod
    def set_video_source(source):
        Recface.video_source = source

    @staticmethod
    def frames():

        camera = cv2.VideoCapture(Recface.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        framex = camera.get(3)
        framey = camera.get(4)
        centerx = framex/2
        centery = framey/2
        font = cv2.FONT_HERSHEY_SIMPLEX

        uploadimages = Uploadimage.objects.filter(rec__icontains="1")
        known_face_encodings = []
        known_face_names = []

        for uploadimage in uploadimages:
            filename = uploadimage.filename
            peoplename = uploadimage.peoplename
            encodingfile = os.path.dirname(os.path.dirname(__file__)) + '/media/recface/file/' + filename + '.npy'
            encodingexit = os.path.exists(encodingfile)
            if encodingexit:
                known_face_names.append(peoplename)
                known_face_encodings.append(np.load(encodingfile))

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

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()


# tensorflow lite
class CameraFollowObject(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CameraFollowObject.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(CameraFollowObject, self).__init__()

    @staticmethod
    def set_video_source(source):
        CameraFollowObject.video_source = source

    @staticmethod
    def frames():



# configure arguments -- begin
        MODEL_NAME = 'Sample_TFlite_model'
        GRAPH_NAME = 'detect.tflite'
        LABELMAP_NAME = 'labelmap.txt'
        min_conf_threshold = float(0.7)
        imW, imH = 640, 320
# configure arguments --end

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
        CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

# Load the Tensorflow Lite model.
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

# Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        # Initialize video stream
        videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        time.sleep(1)

        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:
            # Grab frame from video stream
            frame1 = videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

            # boxes: [ymin, xmin, ymax, xmax]

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            maidongnum = 0
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold) and (scores[i] < 1):
                    # if too large, it will ignore
                    if (boxes[i][3] - boxes[i][1] > 0.5) or (boxes[i][2] - boxes[i][0] > 0.9):
                        continue

                    maidongnum = maidongnum + 1
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                else:
                    break

            if (maidongnum == 1):
                followboxymin = boxes[0][0]
                followboxxmin = boxes[0][1]
                followboxymax = boxes[0][2]
                followboxxmax = boxes[0][3]

                followxc=(followboxxmin + followboxxmax)/2
                followyc=(followboxymin + followboxymax)/2

                print(followxc, followyc)

                if (followxc > 0.5):
                    robot.right(0.3)
                    time.sleep(0.2)
                    robot.stop()
                    print("left")
                else:
                    robot.left(0.3)
                    time.sleep(0.2)
                    robot.stop()
                    print("right")

                bilv = round(followboxxmax - followboxxmin, 2)
                if (bilv > 0.5):
                    robot.stop()
                    print("stop stop")
                else:
                    robot.forward(speed=1-bilv)
                    print("go go go")

            else:
                robot.stop()            
            # Draw framerate in corner of frame
            cv2.putText(frame, str(maidongnum) + ' bottle', (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()



# tensorflow lite
class CameraObjectDetect(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CameraObjectDetect.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(CameraObjectDetect, self).__init__()

    @staticmethod
    def set_video_source(source):
        CameraObjectDetect.video_source = source

    @staticmethod
    def frames():



# configure arguments -- begin
        MODEL_NAME = 'Sample_TFlite_model'
        GRAPH_NAME = 'objectdetect.tflite'
        LABELMAP_NAME = 'labelmapobjectdetect.txt'
        min_conf_threshold = float(0.7)
        imW, imH = 640, 320
# configure arguments --end

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
        CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
        with open(PATH_TO_LABELS, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
        if labels[0] == '???':
            del(labels[0])

# Load the Tensorflow Lite model.
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

        interpreter.allocate_tensors()

# Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        floating_model = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        frame_rate_calc = 1
        freq = cv2.getTickFrequency()

        # Initialize video stream
        videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        time.sleep(1)

        #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        while True:
            t1 = cv2.getTickCount()
            # Grab frame from video stream
            frame1 = videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

            # boxes: [ymin, xmin, ymax, xmax]

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold) and (scores[i] < 1):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))

                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()


def stopstatus():
    os.system("ps -aux | grep dragoncar | cut -d ' ' -f9 | xargs kill -9")
    os.system("ps -aux | grep dragoncar | cut -d ' ' -f10 | xargs kill -9")


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


def detectcarnumber_feed(request):
    return StreamingHttpResponse(gen(CameraDetectcarnumber()),
        content_type='multipart/x-mixed-replace; boundary=frame')


# opencv camera
def followme_feed(request):
    return StreamingHttpResponse(gen(Camera()),
        content_type='multipart/x-mixed-replace; boundary=frame')


# opencv camera
def followobject_feed(request):
    return StreamingHttpResponse(gen(CameraFollowObject()),
        content_type='multipart/x-mixed-replace; boundary=frame')


# opencv camera
def recface_feed(request):
    return StreamingHttpResponse(gen(Recface()),
        content_type='multipart/x-mixed-replace; boundary=frame')


# opencv camera
def photofacepic_feed(request):
    return StreamingHttpResponse(gen(Photofacepic()),
        content_type='multipart/x-mixed-replace; boundary=frame')


# opencv camera
def objectdetect_feed(request):
    return StreamingHttpResponse(gen(CameraObjectDetect()),
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
    global servoudvalue
    global servolrvalue

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
    if "servod" in request.POST:
      if servoudvalue < 1.0:
        servoudvalue = servoudvalue + 0.1
        servoud.value = servoudvalue
    if "servou" in request.POST:
      if servoudvalue > -1.0:
        servoudvalue = servoudvalue - 0.1
        servoud.value = servoudvalue
    if "servlr" in request.POST:
      if servolrvalue < 1.0:
        servolrvalue = servolrvalue + 0.1
        servolr.value = servolrvalue
    if "servlr" in request.POST:
      if servolrvalue > -1.0:
        servolrvalue = servolrvalue - 0.1
        servolr.value = servolrvalue

  return render(request, 'dragoncar/videocar.html')


def followme(request):
  return render(request, 'dragoncar/followme.html')


def voicecontrol(request):
  return render(request, 'dragoncar/voicecontrol.html')


def voicecar(request):
  return render(request, 'dragoncar/voicecar.html')


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


def upload_voicecar(request):
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
        wf = wave.open(BASE_DIR + '/media/voice/voicecar.wav', "rb")

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                rec.Result()

        data = json.loads(rec.FinalResult())
        voicetext = data['text']

        print(voicetext)

        qian = ["前", "前进", "向前", "钱"]
        yizhiqian = ["一直前", "一直前进", "一 值钱", "一直 前进", "以 值钱", "一直 强劲", "一直 前列", "以 之前"]
        hou = ["后", "后退", "向后", "倒退", "向 后", "下 后"]
        yizhihou = ["一直退", "一直 退", "一直 后退", "一直 后"]
        zuo = ["左", "左转", "向左", "左转弯", "着", "走啊", "着 转弯", "向着", "向 走啊", "左手 啊", "左 转"]
        you = ["右", "向右", "右转", "右转弯", "又", "向 右", "右转 啦", "由 转弯", "享有", "有"]
        ting = ["停车", "刹车", "停下", "停"]

        if voicetext in qian:
            print("qianqianqian")
            robot.forward(0.5)
            time.sleep(0.1)
            robot.stop()
        elif voicetext in yizhiqian:
            print("yizhiqian")
            robot.forward(0.5)
        elif voicetext in hou:
            print("hou")
            robot.backward(0.5)
            time.sleep(0.1)
            robot.stop()
        elif voicetext in yizhihou:
            print("yizhihou")
            robot.backward(0.5)
        elif voicetext in zuo:
            print("zuo")
            robot.left(0.5)
            time.sleep(0.1)
            robot.stop()
        elif voicetext in you:
            print("you")
            robot.right(0.5)
            time.sleep(0.1)
            robot.stop()
        elif voicetext in ting:
            print("tingtingting")
            robot.stop()

        return HttpResponse("upload over!")


def pipoweroff():
    os.system("sudo poweroff")


def pirestart():
    os.system("sudo reboot")


def powermanage(request):
    if "pipoweroff" in request.POST:
        pipoweroff()
    if "pirestart" in request.POST:
        pirestart()

    return render(request, 'dragoncar/powermanage.html')


def follow(request):
    return render(request, 'dragoncar/follow.html')


def followobject(request):
  return render(request, 'dragoncar/followobject.html')


def objectdetect(request):
  return render(request, 'dragoncar/objectdetect.html')


def dragonvideo(request):
  return render(request, 'dragoncar/dragonvideo.html')


def detectcarnumber(request):
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

  return render(request, 'dragoncar/detectcarnumber.html')


# detect car number
class CameraDetectcarnumber(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CameraDetectcarnumber.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(CameraDetectcarnumber, self).__init__()

    @staticmethod
    def set_video_source(source):
        CameraDetectcarnumber.video_source = source

    @staticmethod
    def frames():

        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        font = ImageFont.truetype(os.path.dirname(os.path.dirname(__file__)) + '/font/jdjls.ttf', 40)

        while True:
            # read current frame
            _, frame = camera.read()
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            carinfors = HyperLPR_plate_recognition(frame)
            for carinfor in carinfors:
                carnum = carinfor[0]
                caraccuracy = carinfor[1]
                rect = carinfor[2]
                img = cv2.rectangle(frame,(rect[0], rect[1]),(rect[2], rect[3]),(255,0,0),2)
                img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 文字颜色
                fillColor = (255,0,0)
                # 文字输出位置
                position = (rect[0], rect[3])
                # 输出内容
                strc = carnum
                # 需要先把输出的中文字符转换成Unicode编码形式
                if not isinstance(strc, str):
                    strc = strc.decode('utf8')

                draw = ImageDraw.Draw(img_PIL)
                draw.text(position, strc, font=font, fill=fillColor)

                frame = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()


def detectpeopleface(request):
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

  return render(request, 'dragoncar/recface/detectpeopleface.html')


def uploadfacepic(request):
    return render(request, 'dragoncar/recface/uploadfacepic.html')


def upload_face_pic(request):
    if request.method == "POST" and request.POST.get('facename') != "":
        facepic =request.FILES.get("facefile", None)
        if not facepic:
            print("no files for upload!")
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("media/recface/pic",facepic.name),'wb+')
        for chunk in facepic.chunks():
            destination.write(chunk)
        destination.close()
        print("upload over!")

        peoplename = request.POST.get('facename')
        newimage = Uploadimage(filename=facepic.name, filesize=facepic.size, peoplename=peoplename)
        newimage.save()

        return render(request, 'dragoncar/recface/uploadfacepicok.html')
    else:
        return render(request, 'dragoncar/recface/uploadfacepic.html')


def managefacepic(request):
    if request.GET.get('facename') == None:
        return render(request, 'dragoncar/recface/managefacepic.html')

    if "createrecfile" in request.GET:
        createfacepicrecfile()

    facename = request.GET.get('facename')
    uploadimage = Uploadimage.objects.filter(peoplename__icontains=facename)

    return render(request, 'dragoncar/recface/managefacepic.html', {'uploadimage': uploadimage,})


class Photofacepic(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Photofacepic.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Photofacepic, self).__init__()

    @staticmethod
    def set_video_source(source):
        Photofacepic.video_source = source

    @staticmethod
    def frames():

        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()


def photoface(request):
    if "photopicface" in request.POST:
        snapshotphoto()
        print("photo successful")

    return render(request, 'dragoncar/recface/photoface.html')


def snapshotphoto():
    cam = Camera()
    photo = cam.get_frame()
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    file = open(os.path.dirname(os.path.dirname(__file__)) + '/media/recface/pic/' + filename, 'wb+')
    file.write(photo)
    file.close()

    filesize = 60000
    peoplename = "666666"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    newimage = Uploadimage(filename=filename, filesize=filesize, peoplename=peoplename, timestamp=timestamp)
    newimage.save()


def photofacemodify(request):

    if request.GET.get('timestamp') == None:
        return render(request, 'dragoncar/recface/managefacepic.html')

    if "cx" in request.GET:
        timestamp = request.GET.get('timestamp')
        uploadimage = Uploadimage.objects.filter(timestamp__icontains=timestamp)

        return render(request, 'dragoncar/recface/photofacemodify.html', {'uploadimage': uploadimage,})

    if "gx" in request.GET:
        timestamp = request.GET.get('timestamp')
        peoplename = request.GET.get('peoplename')
        rec = request.GET.get('rec')

        Uploadimage.objects.filter(timestamp=timestamp).update(peoplename=peoplename, rec=rec)

        uploadimage = Uploadimage.objects.filter(timestamp__icontains=timestamp)

        return render(request, 'dragoncar/recface/photofacemodifyok.html', {'uploadimage': uploadimage,})


def createfacepicrecfile():
    uploadimages = Uploadimage.objects.filter(rec__icontains="1")
    for uploadimage in uploadimages:
        filename = uploadimage.filename
        encodingfile = os.path.dirname(os.path.dirname(__file__)) + '/media/recface/file/' + filename + '.npy'
        encodingexit = os.path.exists(encodingfile)
        if not encodingexit:
            peopleimage = face_recognition.load_image_file(os.path.dirname(__file__) + '/../media/recface/pic/' + filename)
            face_encoding = face_recognition.face_encodings(peopleimage)[0]
            # np.save(filename, fileencoding)
            np.save(encodingfile, face_encoding)


def qrscan_feed(request):
    return StreamingHttpResponse(gen(CameraQRScan()),
        content_type='multipart/x-mixed-replace; boundary=frame')


def qrscan(request):
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

  return render(request, 'dragoncar/qrsacn.html')


# detect car number
class CameraQRScan(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CameraQRScan.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(CameraQRScan, self).__init__()

    @staticmethod
    def set_video_source(source):
        CameraQRScan.video_source = source

    @staticmethod
    def frames():

        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        font = ImageFont.truetype(os.path.dirname(os.path.dirname(__file__)) + '/font/jdjls.ttf', 40)

        while True:
            # read current frame
            _, frame = camera.read()
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            barcodes = pyzbar.decode(frame)
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect

                img = cv2.rectangle(frame,(x, y), (x + w, y + h),(255,0,0),2)
                img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                text = "Text:" + barcodeData

                # 文字颜色
                fillColor = (255,0,0)
                # 文字输出位置
                position = (10,10)
                # 输出内容
                strc = text

                if not isinstance(strc, str):
                    strc = strc.decode('utf8')

                draw = ImageDraw.Draw(img_PIL)
                draw.text(position, strc, font=font, fill=fillColor)

                frame = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', frame)[1].tobytes()


