import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from threading import Thread
from queue import Queue

import logging
import cv2

import os
import socket

import struct

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

                    

class Visualizer():
    def __init__(self):
        self.LABEL_DICT = {
                0.0: 'bike',
                1.0: 'bus',
                2.0: 'car',
                3.0: 'motor',
                4.0: 'person',
                5.0: 'rider',
                6.0: 'traffic_light',
                7.0: 'ts_priority',
                7.1: 'ts_stop',
                7.2: 'ts_no_entry',
                7.3: 'ts_one_way',
                7.4: 'ts_crossing',
                7.5: 'ts_fw_entry',
                7.6: 'ts_fw_exit',
                7.7: 'ts_parking',
                7.8: 'ts_roundabout',
                8.0: 'train',
                9.0: 'truck',
                10.0:'other'
        }
        self.serverIp   =  os.environ['IP_PC'] # PC ip
        self.port       =  2244            # com port
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

    def get_image_od(self, image_in, object_list, colour):
        # based on https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
        image = image_in.copy()

        #print(object_list)

        if not object_list:
            #print('list was empty')
            return image
        else:
            for obj in object_list:
                #print(obj)
                bbox = obj['bbox']
                xmin = bbox.xmin
                xmax = bbox.xmax
                ymin = bbox.ymin
                ymax = bbox.ymax

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colour, 2)
                y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                label = self.LABEL_DICT.get(obj['id'],"OTHER")
                cv2.putText(image,"{}: {:.2f}%".format(label, obj['score'] * 100),
                            (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)

        return image

    def init_socket(self):
        """Initialize the socket.
        """
        logging.debug("establishing connection to visualization receiver")
        self.client_socket = socket.socket()
        self.connection = None
        # Trying repeatedly to connect the camera receiver.
        while self.connection is None:
            try:
                self.client_socket.connect((self.serverIp, self.port))
                self.client_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                self.connection = self.client_socket.makefile('wb')
            except ConnectionRefusedError as error:
                time.sleep(0.5)
                pass

    def transmit_image(self,image):
        try:
            result, image = cv2.imencode('.jpg', image, self.encode_param)
            data   =  image.tobytes()
            size   =  len(data)
            self.connection.write(struct.pack("<L",size))
            self.connection.write(data)
        except Exception as e:
            logging.debug("failed to stream image"+str(e))
            self.connection = None
            self.init_socket()

    def display_image(self,image):
        cv2.imshow('Image',image)
        cv2.waitKey(1)

class VisualizerThread(Thread):
    def __init__(self,
                    visualizer,
                    inQ_img, inQ_od_ts, inQ_od_others, inQ_ld,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(VisualizerThread,self).__init__()
        self.target = target
        self.name = name
        self.vis = visualizer
        self.out_img_size=(640,480)
        self.prev_transmit_time=time.perf_counter()
        self.inQ_img = inQ_img
        self.inQ_ld = inQ_ld
        self.inQ_od_ts = inQ_od_ts
        self.inQ_od_others = inQ_od_others

    
    def run(self):
        self.vis.init_socket()
        while True:
            if  ((not self.inQ_img.empty())
                and (not self.inQ_od_ts.empty())
                and (not self.inQ_od_others.empty())
                and (not self.inQ_ld.empty())):

                obj_ts = self.inQ_od_ts.get()
                obj_others = self.inQ_od_others.get()
                lanes, intersection, pp_img = self.inQ_ld.get()
                img = self.inQ_img.get()

                img_od_1 = self.vis.get_image_od(img,obj_ts,colour=(255, 0, 0))
                img_od_2 = self.vis.get_image_od(img_od_1,obj_others,colour=(0, 0, 255))
                #logging.debug("going to transmit image with OD visualization")
                img_out = cv2.resize(img_od_2,self.out_img_size)
                
                actual_fps=1.0/((time.perf_counter()-self.prev_transmit_time))
                cv2.putText(img_out,"%.2f FPS"%actual_fps,(0,img_out.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                #self.vis.transmit_image(img_out)
                self.vis.transmit_image(pp_img)
                self.prev_transmit_time=time.perf_counter()
                #self.vis.display_image(img_out)
            time.sleep(0.01)