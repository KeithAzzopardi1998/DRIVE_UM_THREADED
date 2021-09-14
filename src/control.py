import cv2
import numpy as np
from threading import Thread
from queue import Queue
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import time
import math

class MockController():
    def __init__(self):
        pass
    def getCommands(self):
        return [0.0,1.0]

class AutonomousController():
    def __init__(self):
        self.angle_weights = np.array([0.7, 0.2, 0.1])
        self.angles_to_store = 3
        self.last_n_angles = np.zeros(self.angles_to_store)
        self.index = 0
        
        self.actions = ["right",
                        "left",
                        "right",
                        "left",
                        "left",
                        "straight",
                        "right"
                        ]
        self.action_counter = 0

    def getCommands(self, lane_info, intersection_info, obj_info,frame_size, nucleo_queue):
        try:
            #logging.debug("lane_info %s"%str(lane_info))
            #logging.debug("intersection_info %s"%str(intersection_info))
            ld_left, ld_right = lane_info
            int_grad, int_y = intersection_info

            # if int_y >= 320:
            #     self.routine_intersection(int_grad, int_y)
            #logging.debug("calling cruise routine")
            self.routine_cruise(ld_left, ld_right,frame_size, nucleo_queue)
            
        except Exception as e:
            print("AutonomousController failed:\n",e,"\n\n")
    
    def routine_intersection(self,intersection_grad,intersection_y):
        logging.debug("routine_intersection: STOPPING AT INTERSECTION AT Y=%d"%intersection_y)
        logging.debug("the gradient is %d"%intersection_grad)
        '''
        self.car.stop(0.0)
        time.sleep(3)

        #the angle at which we are approaching the intersection (in degrees)
        theta = math.atan(intersection_grad)
        alpha = 90 + math.atan(intersection_grad)
        print("the angle of approach is ",alpha)

        # TODO: how to determine these?
        next_action = self.getNextAction()
        
        if next_action=="right":
            print("routine_intersection: making right turn")
            self.car.drive(0.15, 0.0 + (theta*3))
            time.sleep(3)
            self.car.drive(0.15, 20)
            time.sleep(4)
        elif next_action=="left":
            print("routine_intersection: making left turn")
            self.car.drive(0.15, 0.0 + (theta*3))
            time.sleep(3)
            self.car.drive(0.15, 0.0)
            time.sleep(3)
            self.car.drive(0.15, -20)
            time.sleep(5)
        elif next_action=="straight":
            print("routine_intersection: going straight")
            self.car.drive(0.15, 0.0 + (theta*3))
            time.sleep(3)
            self.car.drive(0.15, 0.0)
            time.sleep(3)
            self.car.drive(0.15, 0.0)
            time.sleep(3)
        else:
            pass
        
        time.sleep(0.1)
        '''

    def getNextAction(self):
        a = self.actions[self.action_counter]
        self.action_counter +=1
        return a

    def routine_cruise(self,lane_left,lane_right,frame_size, nucleo_queue):
        #logging.debug("checpoint 1")
        steering_angle = self.calculate_steering_angle(lane_left,lane_right,frame_size)
        #logging.debug("checpoint 2")
        self.last_n_angles[self.index % self.angles_to_store] = steering_angle

        weighted_angle = 0.0
        #logging.debug("checpoint 3")
        for i in range(self.angles_to_store):
            weighted_angle += self.last_n_angles[(self.index + i + 1) % self.angles_to_store] * self.angle_weights[i]

        #print('weighted angle', weighted_angle)
        logging.debug("weighted angle: %.2f"%weighted_angle)

        self.index += 1
        if self.index % self.angles_to_store == 0 and self.index >= 20:
            self.index = 0

        #self.car.drive(0.15, weighted_angle)
        #time.sleep(0.1)
        #speed maximum = 0.3
        speed_max=0.15
        #varying the speed based on the steering angle
        #weighted_angle: -15 to 15
        speed = abs(weighted_angle) #0 to 15
        speed = speed / 1000.0 #0 to 0.015
        speed = speed_max - speed # (speed_max-0.015) to speed_max
        command = {
            'action' : 'MCTL',
            'speed'  : float(speed),
            'steerAngle' : float(weighted_angle)
        }
        nucleo_queue.put(command)
    
    def calculate_steering_angle(self,lane_left,lane_right,frame_size):
        #convert from lane lines to lane points
        #logging.debug("checpoint 1aa")
        left_lane_pts = self.points_from_lane_coeffs(lane_left,frame_size)
        right_lane_pts = self.points_from_lane_coeffs(lane_right,frame_size)
        #logging.debug("checpoint 1a")
        height, width,_ = frame_size
        x_offset = 0.0
        #logging.debug("checpoint 1b")
        left_x1, left_y1, left_x2, left_y2 = left_lane_pts
        right_x1, right_y1, right_x2, right_y2 = right_lane_pts
        #logging.debug("checpoint 1c")
        left_found = False if (left_x1==0 and left_y1==0 and left_x2==0 and left_y2==0) else True
        #if left_found: print("found left lane")
        right_found = False if (right_x1==0 and right_y1==0 and right_x2==0 and right_y2==0) else True
        #if right_found: print("found right lane")
        #logging.debug("checpoint 1d")
        if left_found and right_found: #both lanes
            cam_mid_offset_percent = 0.02
            mid = int(width/2 * (1 + cam_mid_offset_percent))
            x_offset = (left_x2 + right_x2) / 2 - mid
        elif left_found and not right_found: #left lane only
            x_offset = left_x2 - left_x1
        elif not left_found and right_found: #right lane ony
            x_offset = right_x2 - right_x1
        else: #no lanes detected
            x_offset = 0
        
        #HACK: this was /2 before
        y_offset = float(height/1.8)
        #logging.debug("checpoint 1e")
        steering_angle = math.atan(x_offset / y_offset) #in radians
        steering_angle = steering_angle * 180.0 / math.pi
        steering_angle = np.clip(steering_angle, -15.0, 15.0)
        return steering_angle
    
    def points_from_lane_coeffs(self,line_coefficients,frame_size):
        A = line_coefficients[0]
        b = line_coefficients[1]
        #logging.debug("checpoint 1ab")
        if A==0.00 and b==0.00:
            return [0,0,0,0]
        #logging.debug(str(frame_size))
        height, width,_ = frame_size
        #logging.debug("checpoint 1ac")
        bottom_y = height - 1
        #this should be where the LaneDetector mask ends
        top_y = 0.6 * height
        # y = Ax + b, therefore x = (y - b) / A
        bottom_x = (bottom_y - b) / A
        # clipping the x values
        bottom_x = min(bottom_x, 2*width)
        bottom_x = max(bottom_x, -1*width)
        #logging.debug("checpoint 1ad")
        top_x = (top_y - b) / A
        # clipping the x values
        top_x = min(top_x, 2*width)
        top_x = max(top_x, -1*width)

        return [int(bottom_x), int(bottom_y), int(top_x), int(top_y)]

class AutonomousControllerThread(Thread):
    def __init__(self,
                    autonomousController,
                    inQ_od_ts, inQ_od_others, inQ_ld,
                    outQ_nucleo,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(AutonomousControllerThread,self).__init__()
        self.target = target
        self.name = name
        self.controller = autonomousController

        #input queues
        self.inQ_ld = inQ_ld
        self.inQ_od_ts = inQ_od_ts
        self.inQ_od_others = inQ_od_others
        self.list_inQs = [self.inQ_ld, self.inQ_od_ts, self.inQ_od_others]

        #output queues
        self.outQ_nucleo = outQ_nucleo
        self.list_outQs = [self.outQ_nucleo]
    
    def ready(self):
        in_ready = not any([q.empty() for q in self.list_inQs])
        out_ready = not any([q.full() for q in self.list_outQs])
        return in_ready and out_ready
    
    def nucleoReady(self):
        return self.outQ_nucleo.empty()

    def run(self):
        while True:
            if  self.ready():
                obj_ts = self.inQ_od_ts.get()
                obj_others = self.inQ_od_others.get()
                lanes, intersection, pp_img = self.inQ_ld.get()

                if self.nucleoReady():
                    com = self.controller.getCommands(lanes,intersection,obj_others, pp_img.shape, self.outQ_nucleo)
                
            time.sleep(0.01)   