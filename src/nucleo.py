import cv2
import numpy as np
from threading import Thread
from queue import Queue
import logging
import serial
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
import time
from src.utils.messageConverter import MessageConverter

class NucleoMock():
    def __init__(self):
        pass
    def executeCommands(self,commands):
        logging.debug("executing %s"%str(commands))

class NucleoInterface():
    def __init__(self):
        # comm init       
        self.serialCom = serial.Serial('/dev/ttyACM0',256000,timeout=0.1)
        self.serialCom.flushInput()
        self.serialCom.flushOutput()
        self.messageConverter = MessageConverter()

    def executeCommands(self,commands):
        logging.debug("executing %s"%str(commands))
        command_msg = self.messageConverter.get_command(**commands)
        self.serialCom.write(command_msg.encode('ascii'))

class NucleoInterfaceThread(Thread):
    def __init__(self,
                    nucleoInterface,
                    inQ_controller,
                    group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(NucleoInterfaceThread,self).__init__()
        self.target = target
        self.name = name
        self.nucleo = nucleoInterface

        #input queues
        self.inQ_controller = inQ_controller
    
    def ready(self):
        return not self.inQ_controller.empty()

    def run(self):
        while True:
            if  self.ready():
                com = self.inQ_controller.get()
                self.nucleo.executeCommands(com)
                
            time.sleep(0.01)   