
import random
import time
from motpy import Box, Detection, MultiObjectTracker
from motpy.testing_viz import draw_detection, draw_track
from loguru import logger

def get_miliseconds():
    return int(round(time.time() * 1000))

def read_detections(results, drop_detection_prob: float = 0.0, add_detection_noise: float = 0.0):
    """ parses and converts MOT16 benchmark annotations to known [xmin, ymin, xmax, ymax] format """
    detections = []
    for i in range(len(results)):
    # for _, row in df[df.frame_idx == frame_idx].iterrows():
        if random.random() < drop_detection_prob:
            continue

        box = [results[i]['xmin'], results[i]['ymin'], results[i]['xmax'], results[i]['ymax']]

        if add_detection_noise > 0:
            for i in range(4):
                box[i] += random.uniform(-add_detection_noise, add_detection_noise)

        detections.append(Detection(box=box))
        # print('detection box')
        # print (box)
        # print(Detection(box=box))

    return detections
    # yield frame_idx, detections
    
def track(frame,res,tracker,drop_detection_prob: float = 0.0, add_detection_noise: float = 0.0):
    detections = read_detections(
    res,
    drop_detection_prob=drop_detection_prob,
    add_detection_noise=add_detection_noise)
    # t1 = get_miliseconds()
    active_tracks = tracker.step(detections)
    track_result = []
    for i in range(len(active_tracks)):
        res_dict ={}
        res_dict['label'] = 0
        res_dict['id'] = abs(hash(active_tracks[i].id)) % 2**16
        res_dict['score'] = 1
        res_dict['xmin'] = active_tracks[i].box[0]
        res_dict['ymin'] = active_tracks[i].box[1]
        res_dict['xmax'] = active_tracks[i].box[2]
        res_dict['ymax'] = active_tracks[i].box[3]
        track_result.append(res_dict)
        # print(track_result)
        
    
    
    # print('active_tracks box')
    # print (active_tracks[0].id)
    # print(Detection(box=box))
    
    # ms_elapsed = get_miliseconds() - t1
    # logger.debug('step duration: %dms' % ms_elapsed)
    

    # visualize predictions and tracklets
    # for track in active_tracks:
    #     draw_track(frame, track)  
    # for det in detections:
    #     draw_detection(frame, det)

    return frame,track_result
    
    
