import cv2
import numpy as np
import src.line_preprocessing as pp

from threading import Thread
from queue import Queue
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import time

class LaneDetector():
    def __init__(self):
        self.img_shape = (480, 640)
        #self.img_shape = (1232, 1640)
        self.height = self.img_shape[0]
        self.width = self.img_shape[1]

        #ROI mask
        a = ( 0.00*self.width, 1.00*self.height)# gives a trapezium-like ROI
        b = ( 0.00*self.width, 0.80*self.height)# gives a trapezium-like ROI
        c = ( 0.20*self.width, 0.45*self.height)# gives a trapezium-like ROI
        d = ( 0.80*self.width, 0.45*self.height)# gives a trapezium-like ROI
        e = ( 1.00*self.width, 0.80*self.height)# gives a trapezium-like ROI
        f = ( 1.00*self.width, 1.00*self.height)# gives a trapezium-like ROI
        g = ( 0.90*self.width, 1.00*self.height)# this part covers the car hood
        h = ( 0.85*self.width, 0.90*self.height)# this part covers the car hood
        i = ( 0.15*self.width, 0.90*self.height)# this part covers the car hood
        j = ( 0.10*self.width, 1.00*self.height)# this part covers the car hood
        self.mask_vertices = np.array([[a,b,c,d,e,f,g,h,i,j]], dtype=np.int32)      

        self.alpha = 1.3 #basic contrast control
        self.beta = 0 #basic brightness control

    def getLanes(self, img_in):
        try:
            # Setting Hough Transform Parameters
            rho = 1 # 1 degree
            theta = (np.pi/180) * 1
            threshold = 15
            min_line_length = 20
            max_line_gap = 10

            left_lane_coefficients  = pp.create_coefficients_list()
            right_lane_coefficients = pp.create_coefficients_list()

            previous_left_lane_coefficients = None
            previous_right_lane_coefficients = None

            intersection_y = -1
            # Begin lane detection pipiline
            img = img_in.copy()
            img = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)
            #img = cv2.convertScaleAbs(img,alpha=2.0,beta=30)
            combined_hsl_img = pp.filter_img_hsl(img)
            grayscale_img = pp.grayscale(combined_hsl_img)
            gaussian_smoothed_img = pp.gaussian_blur(grayscale_img, kernel_size=5)
            canny_img = cv2.Canny(gaussian_smoothed_img, 50, 150)
            segmented_img = pp.getROI(canny_img,self.mask_vertices)
            #print("LANE LINES LOG - SEGMENTED IMAGE SUM", np.sum(segmented_img))
            hough_lines = pp.hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
            #print("LANE LINES LOG - HOUGH LINES", hough_lines)

            preprocessed_img = cv2.cvtColor(segmented_img,cv2.COLOR_GRAY2BGR)
            left_lane_lines, right_lane_lines, horizontal_lines = pp.separate_lines(hough_lines, img)

        except Exception as e:
            #return np.array([[0.0,0.0], [0.0,0.0]], img_in
            left_lane_lines = []
            right_lane_lines = []
            horizontal_lines = []
            preprocessed_img = img_in

        #this function returns the y-intercept of the intersection
        #if one is found, else it returns -1
        intersection_info = self.check_for_intersection(horizontal_lines)

        try:
            left_lane_slope, left_intercept = pp.getLanesFormula(left_lane_lines)  
            #print("left slope:",left_lane_slope)      
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [left_lane_slope, left_intercept])
        except Exception as e:
            print("Using saved coefficients for left coefficients", e)
            smoothed_left_lane_coefficients = pp.determine_line_coefficients(left_lane_coefficients, [0.0, 0.0])
            
        try: 
            right_lane_slope, right_intercept = pp.getLanesFormula(right_lane_lines)
            #print("right slope:",right_lane_slope)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [right_lane_slope, right_intercept])
        except Exception as e:
            print("Using saved coefficients for right coefficients", e)
            smoothed_right_lane_coefficients = pp.determine_line_coefficients(right_lane_coefficients, [0.0, 0.0])

        #return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]),intersection_y, preprocessed_img
        return np.array([smoothed_left_lane_coefficients, smoothed_right_lane_coefficients]),intersection_info, preprocessed_img

    def check_for_intersection(self,lines):
        # if there are no horizontal lines, there is definitely no intersection
        if not lines:
            return [-1,-1]

        # to check if there is an intersection, we first calculate
        # the length of each line
        # each line is of the format [x1, y1, x2, y2]
        # we use the difference of x values (instead of pythagoras)
        # as it is faster, and we know that the lines are of a low gradient
        line_lengths = np.array([ abs(l[0] - l[2]) for l in lines])

        # this is the "consensus function" which determines whether
        # there is an intersection or not
        cond1 = (np.mean(line_lengths) >= (self.width/3))
        cond2 = (len(lines)>10)
        cond3 = (len([ l for l in line_lengths if l >=(self.width*0.75)]) 
                        >= len(line_lengths)*0.5)
        detected = cond1 or cond2 or cond3

        if detected:
            print("detected intersection with condition(s) c1: %s, c2: %s, c3: %s"
                    % (cond1,cond2,cond3))
            slope, intercept = pp.getLanesFormula(lines)
            return [slope, intercept]
        else:
            return [-1,-1]     

class LaneDetectorThread(Thread):
    def __init__(self,
                    laneDetector,
                    inQ_img,
                    outQ_vis,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(LaneDetectorThread,self).__init__()
        self.target = target
        self.name = name
        self.detector = laneDetector
        self.inQ_img = inQ_img
        self.outQ_vis = outQ_vis

    def run(self):
        while True:
            #logging.debug("check 1")
            if  ((not self.inQ_img.empty())
                and (not self.outQ_vis.full())):
                #logging.debug("check 2")
                img = self.inQ_img.get()
                ld_info = self.detector.getLanes(img)
                #logging.debug("check 3")
                self.outQ_vis.put(ld_info)
                
            time.sleep(0.01)   