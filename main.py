import sys
sys.path.append('.')

from src.camera import VideoSpoofer, FramePublisherThread
from src.visualization import Visualizer, VisualizerThread
from src.object_detection import ObjectDetectorTrafficSigns, ObjectDetectorOthers, ObjectDetectorThread

from threading import Thread
from queue import Queue

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

QUEUE_SIZE = 10
q_image_od_ts = Queue(QUEUE_SIZE)
q_image_od_others = Queue(QUEUE_SIZE)
q_image_vis = Queue(QUEUE_SIZE)
q_od_ts_vis = Queue(QUEUE_SIZE)
q_od_others_vis = Queue(QUEUE_SIZE)

if __name__ == '__main__':
    logging.debug("going to launch threads")

    vis = Visualizer()
    t_vis = VisualizerThread(
        name = 'Visualizer'
        visualizer = vis,
        inQ_img = q_image_vis,
        inQ_od_ts = q_od_ts_vis,
        inQ_od_others = q_od_others_vis
    )
    t_vis.start()
    logging.debug("started Visualizer")

    vid = VideoSpoofer(video_path='./bfmc2020_online_1.avi')
    t_vid = FramePublisherThread(
        name = 'FramePublisher',
        frame_source = vid,
        outQ_vis = q_image_vis,
        outQ_od_ts = q_image_od_ts,
        outQ_od_others = q_image_od_others
    )
    t_vid.start()
    logging.debug("started FramePublisher")

    od_ts = ObjectDetectorTrafficSigns(
        model_path = 'models/object_detector_quant_4_edgetpu.tflite',
        thresh = 0.2
        device = ':0'
    )
    t_od_ts = ObjectDetectorThread(
        name = 'ObjectDetector_TrafficSigns',
        objectDetector = od_ts
        inQ_img = q_image_od_ts
        outQ_vis = q_od_ts_vis
    )
    t_od_ts.start()
    logging.debug("started ObjectDetector_TrafficSigns")

    od_others = ObjectDetectorOthers(
        model_path = 'models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite',
        thresh = 0.4
        device = ':1'
    )
    t_od_others = ObjectDetectorThread(
        name = 'ObjectDetector_Others',
        objectDetector = od_others
        inQ_img = q_image_od_others
        outQ_vis = q_od_others_vis
    )
    t_od_others.start()
    logging.debug("started ObjectDetector_Others")