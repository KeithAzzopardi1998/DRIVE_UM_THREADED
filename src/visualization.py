import numpy as np
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
    
    # ===================================== LANE DETECTION VISUALIZATION ===============================
    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
        # Copy the passed image
        img_copy = np.copy(img) if make_copy else img

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

        return img_copy

    def trace_lane_line_with_coefficients(self, img, line_coefficients, top_y, make_copy=True):
        A = line_coefficients[0]
        b = line_coefficients[1]
        if A==0.0 and b==0.0:
            img_copy = np.copy(img) if make_copy else img
            return img_copy

        height, width,_ = img.shape
        bottom_y = height - 1
        # y = Ax + b, therefore x = (y - b) / A
        bottom_x = (bottom_y - b) / A
        # clipping the x values
        bottom_x = min(bottom_x, 2*width)
        bottom_x = max(bottom_x, -1*width)

        top_x = (top_y - b) / A
        # clipping the x values
        top_x = min(top_x, 2*width)
        top_x = max(top_x, -1*width)

        new_lines = [[[int(bottom_x), int(bottom_y), int(top_x), int(top_y)]]]
        return self.draw_lines(img, new_lines, make_copy=make_copy)

    def draw_intersection_line(self, img, y_intercept, make_copy=True):
        _, width,_ = img.shape
        line = [[[0, int(y_intercept), width, int(y_intercept)]]]
        return self.draw_lines(img, line,color=[0, 255, 0], make_copy=make_copy)

    def get_image_ld(self, image_in, lane_info, intersection_info):
        intersection_slope = intersection_info[0]
        intersection_y = intersection_info[1]
        left_coefficients = lane_info[0]
        right_coefficients = lane_info[1]
        top_y = image_in.shape[0]*0.45

        lane_img_left = self.trace_lane_line_with_coefficients(image_in, left_coefficients, top_y, make_copy=True)

        if intersection_y == -1:
            lane_img_final = self.trace_lane_line_with_coefficients(lane_img_left, right_coefficients, top_y, make_copy=False)
        else:
            lane_img_both = self.trace_lane_line_with_coefficients(lane_img_left, right_coefficients, top_y, make_copy=True)
            lane_img_final = self.draw_intersection_line(lane_img_both,intersection_y, make_copy=False)

        # image1 * alpha + image2 * beta + lambda
        # image1 and image2 must be the same shape.
        img_with_lane_weight =  cv2.addWeighted(image_in, 0.7, lane_img_final, 0.3, 0.0)

        return img_with_lane_weight

    # ===================================== OBJECT DETECTION VISUALIZATION ===============================
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

                img_ld = self.vis.get_image_ld(img,lanes,intersection)
                img_od_1 = self.vis.get_image_od(img_ld,obj_ts,colour=(255, 0, 0))
                img_od_2 = self.vis.get_image_od(img_od_1,obj_others,colour=(0, 0, 255))
                #logging.debug("going to transmit image with OD visualization")
                img_out = cv2.resize(img_od_2,self.out_img_size)
                
                actual_fps=1.0/((time.perf_counter()-self.prev_transmit_time))
                cv2.putText(img_out,"%.2f FPS"%actual_fps,(0,img_out.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                
                #use this to test the lane preprocessing
                #self.vis.transmit_image(img_out)
                #time.sleep(0.1)
                #self.vis.transmit_image(pp_img)
                #time.sleep(0.1)

                self.vis.transmit_image(img_out)
                self.prev_transmit_time=time.perf_counter()
                #self.vis.display_image(img_out)
            time.sleep(0.01)