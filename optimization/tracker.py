import math
import numpy as np
import caffe
import cv2
from bounding_box import BBox
import time

class Tracker:

    def __init__(self, context_factor =2):
        self.net = caffe.Net('/home/coline/packages/GOTURN/nets/tracker.prototxt',
                             '/home/coline/packages/GOTURN/nets/models/pretrained_model/tracker.caffemodel',
                             caffe.TEST)
        caffe.set_mode_gpu()
        self.context_factor = context_factor
        self.all_frames = None
        transformer = caffe.io.Transformer({'image':self.net.blobs['image'].data.shape,
                                            'target':self.net.blobs['target'].data.shape})

        # self.img_mean = np.array([104, 117,123])
        # self.img_transpose = (2,0,1)
        transformer.set_mean('image',np.array([104, 117,123] ))
        transformer.set_mean('target', np.array([104, 117,123]))
        transformer.set_transpose('image', (2,1,0))
        transformer.set_transpose('target', (2,1,0))
        transformer.set_transpose('image', (2,0,1))
        transformer.set_transpose('target', (2,0,1))
        # transformer.set_channel_swap('target', (2,1,0))
        # transformer.set_channel_swap('image', (2,1,0))
        # transformer.set_raw_scale('data', 255.0)
        self.transformer = transformer
        # pass
    def load_video(self, directory):
        """ directory is like '/home/coline/test/video1' """
        self.current_dir = directory
        self.current_frame = 0
        # with open(directory+'/groundtruth.txt') as f:
        #     out = f.readline()
        #     coords = out.split(',')
        #     coords[-1]= coords[-1][:-1]
        #     coords = [float(c) for c in coords]
        #     self.gtruth = BBox(self.context_factor, coords=coords)

    def load_full_video(self, directory):
        self.current_dir = directory
        self.current_frame = 0
        frame = self.next_frame()
        if frame is None:
            frame= self.next_frame()
        frames = [frame]
        while frame is not None:
            frame= self.next_frame()
            frames.append(frame)
        self.all_frames = frames
        self.current_frame=0

    def next_loaded_frame(self):
        im = self.all_frames[self.current_frame]
        self.current_frame += 1
        return im

    def next_frame(self):
        filename = self.current_dir + '/{:08d}.jpg'.format(self.current_frame)
        image = cv2.imread(filename)
        self.current_frame += 1

        return image

    def init_tracking(self):
        self.current_frame = 0
        if self.all_frames is not None:
            frame = self.next_loaded_frame()
        else:
            frame = self.next_frame()
            if frame is None:
                frame= self.next_frame()
        self.first_frame = frame
        crop = self.gtruth.crop_frame(frame)
        return crop 

    def preprocess(self, image):
        image = image-self.img_mean
        image = np.transpose(image, self.img_transpose)
        return image

    def regress(self, next_crop, prev_crop):
        h,w,c = next_crop.shape
        h,w,c = prev_crop.shape
        time0 = time.clock()
        filename = 'gym_out.png'
        # tnext_crop = cv2.imread(filename)
        # tprev_crop = cv2.imread('gym_target.png')
        self.net.blobs['image'].data[...] = self.transformer.preprocess('image',next_crop)
        self.net.blobs['target'].data[...] = self.transformer.preprocess('target',prev_crop)
        self.net.blobs['bbox'].data[...] = np.zeros((1,4,1,1)) + 100219
        time1 = time.clock()
        self.net.forward()
        time2 = time.clock()
        result = np.array(self.net.blobs['fc8'].data).squeeze()
        # result = np.array([2.551414, 2.579429, 7.668951, 7.468904])
        # print result
        # import IPython
        # IPython.embed()
        # print time1-time0, time2-time1
        return result / 10.


    def track(self, directory,initial, video=None, boxes=None, video_nodraw=None):
        test=False
        next_frame_func = self.next_frame
        if self.all_frames is not None:
            next_frame_func = self.next_loaded_frame
        self.load_video(directory)
        bbox = initial
        self.gtruth = bbox
        target = self.init_tracking()
        prev_frame = self.first_frame

        i = 0
        if video is not None:
            video = []
        if boxes is not None:
            boxes = []
        test = False
        while True:
            time0 = time.clock()
            # print i#, self.current_frame
            frame = next_frame_func()
            # if i == 0:
            #     import IPython; IPython.embed()
            if frame is None:
                print "end of movie"
                break
            # loose_bbox is pad_image_loc
            target, target_bbox, _,_ = bbox.cropPadImage(prev_frame)
            next_crop, loose_bbox, edge_x, edge_y = bbox.cropPadImage(frame)
            if i == 0:
                if video is not None:
                    video.append(loose_bbox.draw_frame(bbox.draw_frame(self.first_frame, 0),1))
                    #pass
            # import IPython
            # IPython.embed()

            points = self.regress(next_crop, target)
            bbox = loose_bbox.get_tight_bbox(points, next_crop,test, frame, edge_x, edge_y)
            target = bbox.crop_frame(frame)
            prev_frame = frame
            i+=1 
            if video_nodraw is not None:
                video_nodraw.append(frame)
            if video is not None:
                video.append(loose_bbox.draw_frame(bbox.draw_frame(frame, 0), 1))
                #video.append(next_crop)
            if boxes is not None:
                boxes.append(bbox)
        return video, boxes, video_nodraw

def animate(images, name):
    import moviepy.editor as mpy
    leng = len(images)
    def make_frame(n):
        tmp = images[n,:,:,:]
        return tmp
    clip = mpy.ImageSequenceClip([images[i] for i in range(leng)], fps=20)
    clip.write_gif("/home/coline/visual_features/tracking/videos/"+name+".gif",fps=20)
    return clip


if __name__ == "__main__":
    tracker = Tracker()

    #hand2 has problem at 201, ball 448x
    for obj in ['ball', 'fish1', 'fernando', 'gymnastics', 'drunk', 'diving', 'skating', 'sphere', 'jogging', 'polarbear', 'motocross']:
        print "----------------------- obj:",obj," ------------------"
        directory = '/home/coline/packages/GOTURN/vot2014/'+obj
        with open(directory+'/groundtruth.txt') as f:
            out = f.readline()
            coords = out.split(',')
            coords[-1]= coords[-1][:-1]
            coords = [float(c) for c in coords]
            gtruth = BBox(tracker.context_factor, coords=coords)
        video,_, _ = tracker.track('/home/coline/packages/GOTURN/vot2014/'+obj, initial=gtruth,video =[])
        for f in range(len(video)):
            # print f
            cv2.imwrite("/home/coline/visual_features/tracking/videos/"+'vot_'+obj+'/'+str(f)+'.jpg', video[f])
        print "video was", len(video), "long"
        # import IPython
        # IPython.embed()
        animate(video, 'vot_'+obj)

