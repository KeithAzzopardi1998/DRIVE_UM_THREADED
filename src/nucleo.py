import cv2
import numpy as np
from threading import Thread
from queue import Queue
import logging
import serial
import time
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
        self.resetMotor()

    def resetMotor(self):
        cmd = {
            'action' : 'BRAK',
            'steerAngle' : float(0.0)
        }
        self.executeCommand(cmd)

    def executeCommand(self,command):
        logging.debug("executing %s"%str(command))
        #check if it is one of our custom commands
        action = command['action']
        if action=="NOOP":
            pass
        elif action=="WAIT":
            time.sleep(command['duration'])
        else: #nucleo command
            command_msg = self.messageConverter.get_command(**command)
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
                self.nucleo.executeCommand(com)
                
            time.sleep(0.01)   