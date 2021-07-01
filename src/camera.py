import cv2
import time
from threading import Thread
from queue import Queue

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


class FramePublisherThread(Thread):
    def __init__(self,
                    frame_source,
                    outQ_vis, outQ_od_ts, outQ_od_others ,
                    fps,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(FramePublisherThread,self).__init__()
        self.target = target
        self.name = name
        self.source = frame_source
        self.fps = fps
        self.outQ_vis = outQ_vis
        self.outQ_od_ts = outQ_od_ts
        self.outQ_od_others = outQ_od_others

    def run(self):
        while True:
            if (not self.outQ_vis.full()) and (not self.outQ_od_ts.full()) and (not self.outQ_od_others.full()):
                # img = self.img
                img = self.source.get_frame()
                #logging.debug("pushing image to queue")
                self.outQ_vis.put(img)#BGR
                self.outQ_od_ts.put(img)#BGR
                self.outQ_od_others.put(img)#BGR
                time.sleep(1.0/self.fps)