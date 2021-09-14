import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue
import io
import picamera

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

class VideoSpoofer():
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def get_frame(self):
        ret, img = self.cap.read()
        if ret:
            return img
        else:
            logging.error("Failed to read video frame")

class Camera():
    def __init__(self):
        self.width=320
        self.height=240
        self.camera = picamera.PiCamera()
        self.camera.resolution = (self.width,self.height)
        self.camera.framerate = 5

    def get_frame(self):
        out_img = np.empty((self.height*self.width*3,), dtype=np.uint8) 
        self.camera.capture(out_img,'bgr')
        return out_img.reshape((self.height,self.width,3))
    
class FramePublisherThread(Thread):
    def __init__(self,
                    frame_source,
                    outQ_vis, outQ_od_ts, outQ_od_others, outQ_ld,
                    fps,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(FramePublisherThread,self).__init__()
        self.target = target
        self.name = name
        self.source = frame_source
        self.fps = fps
        self.outQ_vis = outQ_vis
        self.outQ_ld = outQ_ld
        self.outQ_od_ts = outQ_od_ts
        self.outQ_od_others = outQ_od_others
        #self.out_size = (480, 640)
        self.out_size = (640, 480)

    def run(self):
        while True:
            if  ((not self.outQ_vis.full())
                and (not self.outQ_od_ts.full())
                and (not self.outQ_od_others.full())
                and (not self.outQ_ld.full())):
                # img = self.img
                img = self.source.get_frame()
                img = cv2.resize(img, self.out_size, interpolation = cv2.INTER_AREA)
                #logging.debug("pushing image to queue")
                self.outQ_vis.put(img)#BGR
                self.outQ_ld.put(img)#BGR
                self.outQ_od_ts.put(img)#BGR
                self.outQ_od_others.put(img)#BGR
                time.sleep(1.0/self.fps)

class CameraThread(Thread):
    def __init__(self,
                    outQ_vis, outQ_od_ts, outQ_od_others, outQ_ld,
                    fps,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(CameraThread,self).__init__()
        self.target = target
        self.name = name
        self.fps = fps
        self.outQ_vis = outQ_vis
        self.outQ_ld = outQ_ld
        self.outQ_od_ts = outQ_od_ts
        self.outQ_od_others = outQ_od_others
        self.list_outQs = [self.outQ_vis,self.outQ_ld,self.outQ_od_ts,self.outQ_od_others]
        #self.out_size = (480, 640)
        self.imgSize = (640, 480)
    
        self._stream      =   io.BytesIO()

    def _init_camera(self):
        """Init the PiCamera and its parameters
        """
        
        # this how the firmware works.
        # the camera has to be imported here
        from picamera import PiCamera

        # camera
        self.camera = PiCamera()

        # camera settings
        self.camera.resolution      =   self.imgSize
        self.camera.framerate       =   self.fps
        self.camera.brightness      =   80 #0-100
        self.camera.shutter_speed   =   1200
        self.camera.contrast        =   85 #-100 - 100
        self.camera.iso             =   0 # auto
        self.recordMode             =   False

    def ready(self):
        out_ready = not any([q.full() for q in self.list_outQs])
        return out_ready
    
    def _streams(self):
        """Stream function that actually published the frames into the pipes. Certain 
        processing(reshape) is done to the image format. 
        """  
        while True:   
            yield self._stream
            self._stream.seek(0)
            data = self._stream.read()

            # read and reshape from bytes to np.array
            data  = np.frombuffer(data, dtype=np.uint8)
            data  = np.reshape(data, (480, 640, 3))
            self.outQ_vis.put(data)#BGR
            self.outQ_ld.put(data)#BGR
            self.outQ_od_ts.put(data)#BGR
            self.outQ_od_others.put(data)#BGR          
            self._stream.seek(0)
            self._stream.truncate()

    def run(self):
        """Start sending data through pipe. 
        """
        self._init_camera()
        
        self.camera.capture_sequence(
                                    self._streams(), 
                                    use_video_port  =   True, 
                                    format          =   'bgr')
     