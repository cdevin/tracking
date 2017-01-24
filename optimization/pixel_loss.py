import math
import numpy as np
import caffe
import cv2
from tracker import Tracker, animate
from bounding_box import BBox
import random

random.seed(10)

def gen_initial(tracker):
    frame = tracker.all_frames[0]
    shape = frame.shape
    a = random.randint(0, shape[0]-50)
    b = random.randint(a+10, shape[0])
    c = random.randint(0, shape[1]-50)
    d = random.randint(c+10, shape[1])
    y1 = min(a,b)
    y2 = max(a,b)
    x1 = min(c,d)
    x2 = max(c,d)
    init = BBox(2)
    print [x1,x2,y1,y2]
    init.set_xy(x1,x2,y1,y2)

    return init

def pixel_error(tracker, boxes, initial):
    frames = tracker.all_frames[:50]
    shape = frames[0].shape
    first_crop = initial.crop_frame(frames[0])
    # import IPython
    # IPython.embed()
    crop = first_crop.shape
    avg_first = np.mean(first_crop.reshape(crop[0]*crop[1], crop[2]),axis=0)
    error = 0
    total_pixels = shape[0]*shape[1]
    for f in range(1,len(frames)):
        # print "frame", f
        for i in range(shape[0]):
            for j in range(shape[1]):
                if boxes[f-1].in_box(i,j):
                    # weight by std?
                    error+= np.linalg.norm(frames[f][i,j,:]-avg_first)/total_pixels
                elif initial.in_box(i,j):
                    error+=0
                else:
                    error+= np.linalg.norm(frames[f][i,j,:]-frames[0][i,j,:])/total_pixels
    print "error", error
    return error

obj = 'gymnastics'
directory = '/home/coline/packages/GOTURN/vot2014/'+obj
tracker = Tracker()
print "about to load video"
tracker.load_full_video(directory)
print "loaded video"
initial = gen_initial(tracker)

# Todo: wirte function that generates boxes from close by
for i in range(1):
    best_init = initial,0
    best_error = float('inf')
    for j in range(100):
        print "STARTING", j
        video, boxes,_ = tracker.track('/home/coline/packages/GOTURN/vot2014/'+obj, initial,video =[], boxes=[], video_nodraw=[])
        animate(video, 'ball_test/vot_'+obj+'_'+str(j))
        #error = pixel_error(tracker, boxes, initial)
        # if error < best_error:
        #     print "prev best was", best_error, "now best is", error
        #     best_error = error
        #     best_init = initial,j
        # print error
        initial = gen_initial(tracker)

    print "BEST WAS", best_init[1]
