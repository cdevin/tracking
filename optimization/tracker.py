import math
import numpy as np
import caffe
import cv2
from bounding_box import BBox


class Tracker:

    def __init__(self, context_factor =2):
        self.net = caffe.Net('/home/coline/packages/GOTURN/nets/tracker.prototxt',
                             '/home/coline/packages/GOTURN/nets/models/pretrained_model/tracker.caffemodel',
                             caffe.TEST)
        self.context_factor = context_factor
        transformer = caffe.io.Transformer({'image':self.net.blobs['image'].data.shape,
                                            'target':self.net.blobs['target'].data.shape})

        # self.img_mean = np.array([104, 117,123])
        # self.img_transpose = (2,0,1)
        transformer.set_mean('image',np.array([104, 117,123] ))
        transformer.set_mean('target', np.array([104, 117,123]))
        transformer.set_transpose('image', (2,0,1))
        transformer.set_transpose('target', (2,0,1))
        # transformer.set_channel_swap('data', (2,1,0))
        # transformer.set_raw_scale('data', 255.0)
        self.transformer = transformer
        # pass
    def load_video(self, directory):
        """ directory is like '/home/coline/test/video1' """
        self.current_dir = directory
        self.current_frame = 0
        with open(directory+'/groundtruth.txt') as f:
            out = f.readline()
            coords = out.split(',')
            coords[-1]= coords[-1][:-1]
            coords = [float(c) for c in coords]
            self.gtruth = BBox(self.context_factor, coords=coords)

    def next_frame(self):
        filename = self.current_dir + '/{:08d}.jpg'.format(self.current_frame)
        image = cv2.imread(filename)
        self.current_frame += 1

        return image

    def init_tracking(self):
        self.current_frame = 0
        frame = self.next_frame()
        if frame is None:
            frame= self.next_frame()
        crop = self.gtruth.crop_frame(frame)
        return crop 

    def preprocess(self, image):
        image = image-self.img_mean
        image = np.transpose(image, self.img_transpose)
        return image

    def regress(self, next_crop, prev_crop):

        h,w,c = next_crop.shape
        # self.net.blobs['image'].reshape(1,c,h,w)
        # self.net.blobs['image'].data[...] = [self.preprocess(next_crop)]
        h,w,c = prev_crop.shape
        # self.net.blobs['target'].reshape(1,c,h,w)
        # self.net.blobs['target'].data[...] = [self.preprocess(prev_crop)]

        # transformer = caffe.io.Transformer({'image':self.net.blobs['image'].data.shape,
        #                                     'target':self.net.blobs['target'].data.shape})
        # transformer.set_mean('image', np.array([104, 117,123]))
        # transformer.set_mean('target', np.array([104, 117,123]))
        # transformer.set_transpose('image', (2,0,1))
        # transformer.set_transpose('target', (2,0,1))
        # # transformer.set_channel_swap('data', (2,1,0))
        # # transformer.set_raw_scale('data', 255.0)
        # self.transformer = transformer
        self.net.blobs['image'].data[...] = self.transformer.preprocess('image',next_crop)
        self.net.blobs['target'].data[...] = self.transformer.preprocess('target',prev_crop)

        self.net.forward()
        result = np.array(self.net.blobs['fc8'].data).squeeze()
        return result / 10.


    def track(self, directory, video=None):
        self.load_video(directory)
        bbox = self.gtruth
        target = self.init_tracking()
        i = 0
        if video is not None:
            video = []
        while True:
            frame = self.next_frame()
            if frame is None:
                break
            next_crop, loose_bbox = bbox.get_next_crop(frame)
            points = self.regress(next_crop, target)
            bbox = loose_bbox.get_tight_bbox(points, next_crop)
            target = bbox.crop_frame(frame)
            # if i % 20 ==0:
            #     import IPython
            #     IPython.embed()
            i+=1 
            if video is not None:
                video.append(bbox.draw_frame(frame))
        return video

def animate(images, name):
    import moviepy.editor as mpy
        
    def make_frame(n):
        tmp = images[n,:,:,:]
        return tmp
    clip = mpy.ImageSequenceClip([images[i] for i in range(100)], fps=20)
    clip.write_gif("/home/coline/visual_features/tracking/videos/"+name+".gif",fps=20)
    return clip


tracker = Tracker()
#tracker.load_video('/home/coline/packages/GOTURN/vot2014/ball')
#tracker.init_tracking()
# frame = tracker.next_frame()
# frame2 = tracker.next_frame()
# target = tracker.gtruth.crop_frame(frame)
# image = tracker.gtruth.get_next_crop(frame2)
#tracker.regress(image, target)  #
for obj in ['hand2', 'ball', 'fish1', 'fernando', 'gymnastics', 'drunk', 'driving', 'skating', 'sphere']:
    video = tracker.track('/home/coline/packages/GOTURN/vot2014/'+obj, video =[])
    animate(video, 'vot_'+obj)

