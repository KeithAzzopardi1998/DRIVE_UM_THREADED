import cv2
import numpy as np
from threading import Thread
from queue import Queue
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import time

class AutonomousController():
    def __init__(self):
        pass
    def getCommands(self):
        return [0.0,1.0]

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

    def run(self):
        while True:
            if  self.ready():
                obj_ts = self.inQ_od_ts.get()
                obj_others = self.inQ_od_others.get()
                lanes, intersection, pp_img = self.inQ_ld.get()

                com = self.controller.getCommands()
                self.outQ_nucleo.put(com)
                
            time.sleep(0.01)   