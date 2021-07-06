from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from threading import Thread
from queue import Queue

import time
import tflite_runtime.interpreter as tflite

import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import cv2

class ObjectDetectorTrafficSigns():
    def __init__(self,model_path_detect,thresh_detect,model_path_recog,thresh_recog,device):
        #Traffic Sign Detection interpreter
        self.detect_interpreter = make_interpreter(model_path_detect,device=device)
        self.detect_interpreter.allocate_tensors()      
        self.detect_threshold = thresh_detect

        #Traffic Sign Recognition interpreter
        self.tsr_interpreter = tflite.Interpreter(model_path=model_path_recog)
        self.tsr_interpreter.allocate_tensors()
        self.tsr_input_details = self.tsr_interpreter.get_input_details()
        self.tsr_output_details = self.tsr_interpreter.get_output_details()
        self.tsr_threshold = thresh_recog     


    def detect(self,image_pil,image_opencv):
        _, scale = common.set_resized_input(
                        self.detect_interpreter,
                        image_pil.size,
                        lambda size: image_pil.resize(size, Image.ANTIALIAS))

        start = time.perf_counter()
        self.detect_interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(self.detect_interpreter, self.detect_threshold, scale)
        objs_dict = []
        for o in objs:
            if o.id==7:
                temp_o = {}
                #temp_o['id'] = 7.0
                #get the score and bounding box from the detection model
                temp_o['score'] = o.score
                temp_o['bbox'] = o.bbox
                #find the label by running the recognition model
                roi = image_opencv[o.bbox.ymin:o.bbox.ymax, o.bbox.xmin:o.bbox.xmax]
                temp_o['id'] = self.recognize(roi)
                objs_dict.append(temp_o)
        #logging.debug("Inference time: %.2f ms" % (inference_time * 1000))
        return objs_dict

    def set_input_tsr(self, image):
        """Sets the input tensor."""
        tensor_index = self.tsr_input_details[0]['index']
        input_tensor = self.tsr_interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def recognize(self, image):
        input_shape = self.tsr_input_details[0]['shape']
        _, height, width, _ = input_shape

        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        resized_image = resized_image[np.newaxis, :]

        self.set_input_tsr(resized_image)

        self.tsr_interpreter.invoke()

        output_proba = self.tsr_interpreter.get_tensor(self.tsr_output_details[0]['index'])[0]
        tsr_class = np.argmax(output_proba)

        return float("7.%d"%tsr_class)        


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

    def detect(self,image_pil,image_opencv):
        _, scale = common.set_resized_input(
                        self.interpreter,
                        image_pil.size,
                        lambda size: image_pil.resize(size, Image.ANTIALIAS))

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
                img_pil = Image.fromarray(img_opencv)
                #logging.debug("going to start detection")
                objs = self.od_model.detect(img_pil,img_opencv)
                #self.od_model.print_detections(objs)
                self.outQ_vis.put(objs)
                #logging.debug("finished detection ... returning %d objects"%len(objs))
            time.sleep(0.01)        