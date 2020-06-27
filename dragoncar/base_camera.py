import picamera, threading, io, time
from _thread import get_ident
class CameraEvent(object):
        def __init__(self):
                self.events = {}
        def wait(self):
                ident = get_ident()
                if ident not in self.events:
                        self.events[ident] = [threading.Event(), time.time()]
                # print(2)
                return self.events[ident][0].wait()
        def set(self):
                now = time.time()
                remove = None
                for ident, event in self.events.items():
                        if not event[0].isSet():
                                event[0].set()
                                event[1] = now
                        else:
                                if now - event[1] > 5:
                                        remove = ident
                if remove:
                        del self.events[remove]
        def clear(self):
                # print(3)
                self.events[get_ident()][0].clear()
class BaseCamera(object):
        thread = None  
        frame = None  
        last_access = 0  
        event = CameraEvent()
        def __init__(self):
                if BaseCamera.thread is None:
                        BaseCamera.thread = threading.Thread(target=self._thread)
                        BaseCamera.thread.start()
                        while self.get_frame() is None:
                                time.sleep(0)
        def get_frame(self):
                BaseCamera.last_access = time.time()
                BaseCamera.event.wait() 
                BaseCamera.event.clear()
                return BaseCamera.frame
        @staticmethod
        def frames():
                raise RuntimeError('Must be implemented by subclasses.')
        @classmethod
        def _thread(cls):
                print('Starting camera thread.')
                frames_iterator = cls.frames()
                for frame in frames_iterator:
                        BaseCamera.frame = frame
                        BaseCamera.event.set()
                        # print(1)
                        time.sleep(0)
                        if time.time() - BaseCamera.last_access > 10:  
                                frames_iterator.close()
                                print('Stopping camera thread due to inactivity.')
                                break
                BaseCamera.thread = None
class Camera(BaseCamera):
        @staticmethod
        def frames():
                with picamera.PiCamera() as camera:
                        camera.resolution = (320,240)
                        time.sleep(2)
                        stream = io.BytesIO()
                        for foo in camera.capture_continuous(stream, 'jpeg',use_video_port=True):
                                stream.seek(0)
                                yield stream.read()
                                stream.seek(0)
                                stream.truncate()
