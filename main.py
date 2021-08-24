import sys
sys.path.append('.')

from src.camera import VideoSpoofer, Camera, FramePublisherThread
from src.visualization import Visualizer, VisualizerThread
from src.object_detection import ObjectDetectorTrafficSigns, ObjectDetectorOthers, ObjectDetectorThread
from src.lane_detection import LaneDetector, LaneDetectorThread
from src.control import AutonomousController, AutonomousControllerThread
from src.nucleo import NucleoInterface, NucleoInterfaceThread

from threading import Thread
from queue import Queue

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

QUEUE_SIZE = 10
q_image_od_ts = Queue(QUEUE_SIZE)
q_image_od_others = Queue(QUEUE_SIZE)
q_image_vis = Queue(QUEUE_SIZE)
q_image_ld = Queue(QUEUE_SIZE)
q_ld_vis = Queue(QUEUE_SIZE)
q_od_ts_vis = Queue(QUEUE_SIZE)
q_od_others_vis = Queue(QUEUE_SIZE)
q_ld_con = Queue(QUEUE_SIZE)
q_od_ts_con = Queue(QUEUE_SIZE)
q_od_others_con = Queue(QUEUE_SIZE)
q_con_nuc = Queue(QUEUE_SIZE)

if __name__ == '__main__':
    logging.debug("going to launch threads")

    vis = Visualizer()
    t_vis = VisualizerThread(
        name = 'Visualizer',
        visualizer = vis,
        inQ_img = q_image_vis,
        inQ_od_ts = q_od_ts_vis,
        inQ_od_others = q_od_others_vis,
        inQ_ld = q_ld_vis
    )
    t_vis.start()
    logging.debug("started Visualizer")

    #vid = VideoSpoofer(
    #    video_path='./bfmc2020_online_1.avi'
    #)
    vid = Camera()
    t_vid = FramePublisherThread(
        name = 'FramePublisher',
        frame_source = vid,
        fps = 5,
        outQ_vis = q_image_vis,
        outQ_od_ts = q_image_od_ts,
        outQ_od_others = q_image_od_others,
        outQ_ld = q_image_ld
    )
    t_vid.start()
    logging.debug("started FramePublisher")

    od_ts = ObjectDetectorTrafficSigns(
        model_path_detect = 'models/object_detector_quant_4_edgetpu.tflite',
        thresh_detect = 0.3,
        model_path_recog = 'models/object_recognition_quant.tflite',
        thresh_recog = 0.3,
        device = ':0',
    )
    t_od_ts = ObjectDetectorThread(
        name = 'ObjectDetector_TrafficSigns',
        objectDetector = od_ts,
        inQ_img = q_image_od_ts,
        outQ_vis = q_od_ts_vis,
    )
    t_od_ts.start()
    logging.debug("started ObjectDetector_TrafficSigns")

    od_others = ObjectDetectorOthers(
        model_path = 'models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
        thresh = 0.5,
        device = ':1',
    )
    t_od_others = ObjectDetectorThread(
        name = 'ObjectDetector_Others',
        objectDetector = od_others,
        inQ_img = q_image_od_others,
        outQ_vis = q_od_others_vis,
    )
    t_od_others.start()
    logging.debug("started ObjectDetector_Others")

    ld = LaneDetector()
    t_ld =LaneDetectorThread(
        name = 'LaneDetector',
        laneDetector = ld,
        inQ_img = q_image_ld,
        outQ_vis = q_ld_vis
    )
    t_ld.start()
    logging.debug("started LaneDetector")

    con = AutonomousController()
    t_con = AutonomousControllerThread(
        name = 'AutonomousController',
        autonomousController = con,
        inQ_od_ts = q_od_ts_con,
        inQ_od_others = q_od_others_con,
        inQ_ld = q_ld_con,
        outQ_nucleo = q_con_nuc
    )
    t_con.start()
    logging.debug("started AutonomousController")

    nuc = NucleoInterface()
    t_nuc = NucleoInterfaceThread(
        name = 'NucleoInterface',
        nucleoInterface = nuc,
        inQ_controller = q_con_nuc
    )
    t_nuc.start()
    logging.debug("started NucleoInterface")

