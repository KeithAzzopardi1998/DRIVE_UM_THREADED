from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from threading import Thread
from queue import Queue

import time

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import cv2

class ObjectDetectorTrafficSigns():
    def __init__(self,model_path,thresh,device):
        self.interpreter = make_interpreter(model_path,device=device)
        self.interpreter.allocate_tensors()      
        self.threshold = thresh


    def detect(self,image):
        _, scale = common.set_resized_input(
                        self.interpreter,
                        image.size,
                        lambda size: image.resize(size, Image.ANTIALIAS))

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(self.interpreter, self.threshold, scale)
        objs_dict = []
        for o in objs:
            if o.id==7:
                temp_o = {}
                temp_o['id'] = 7.0
                temp_o['score'] = o.score
                temp_o['bbox'] = o.bbox
                objs_dict.append(temp_o)
        #logging.debug("Inference time: %.2f ms" % (inference_time * 1000))
        return objs_dict

class ObjectDetectorOthers():
    def __init__(self,model_path,thresh,device):
        self.interpreter = make_interpreter(model_path,device=device)
        self.interpreter.allocate_tensors()      
        self.threshold = thresh
        # this is used to convert the COCO labels
        # to BDD100k labels. It only contains the
        # classes we are interested in, so if the
        # class is not found here, we label the object
        # using class 10.0, which represents OTHER
        self.coco_to_bdd100k = {
            0 : 4.0, #person
            1 : 0.0, #bike
            2 : 2.0, #car
            3 : 3.0, #motorcycle
            9 : 6.0, #traffic light
        }

    def detect(self,image):
        _, scale = common.set_resized_input(
                        self.interpreter,
                        image.size,
                        lambda size: image.resize(size, Image.ANTIALIAS))

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(self.interpreter, self.threshold, scale)
        objs_dict = []
        for o in objs:
            temp_o = {}
            temp_o['id'] = self.coco_to_bdd100k.get(o.id,10.0)
            temp_o['score'] = o.score
            temp_o['bbox'] = o.bbox
            objs_dict.append(temp_o)
        #logging.debug("Inference time: %.2f ms" % (inference_time * 1000))
        return objs_dict

class ObjectDetectorThread(Thread):
    def __init__(self,
                    objectDetector,
                    inQ_img,
                    outQ_vis,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(ObjectDetectorThread,self).__init__()
        self.target = target
        self.name = name
        self.od_model = objectDetector
        self.inQ_img = inQ_img
        self.outQ_vis = outQ_vis

    def run(self):
        while True:
            if not self.inQ_img.empty():
                img_opencv = self.inQ_img.get()
                img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_opencv)
                #logging.debug("going to start detection")
                objs = self.od_model.detect(img)
                #self.od_model.print_detections(objs)
                self.outQ_vis.put(objs)
                #logging.debug("finished detection ... returning %d objects"%len(objs))
            time.sleep(0.01)        