import cv2
import numpy as np
from threading import Thread
from queue import Queue
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import time

class NucleoInterface():
    def __init__(self):
        pass
    def executeCommands(commands):
        logging.debug("executing %s"%str(commands))

class NucleoInterfaceThread(Thread):
    def __init__(self,
                    nucleoInterface,
                    inQ_controller
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(NucleoInterfaceThread,self).__init__()
        self.target = target
        self.name = name
        self.nucleo = nucleoInterface

        #input queues
        self.inQ_controller = inQ_controller
    
    def ready():
        return not self.inQ_controller.empty()

    def run(self):
        while True:
            if  self.ready():
                com = self.inQ_controller.get()
                self.nucleo.executeCommands(com)
                
            time.sleep(0.01)   