import math
import numpy as np
import caffe
import cv2

class BBox:
    def __init__(self, context_factor, coords=None):
        if coords is not None:
            Ax,Ay,Bx,By,Cx,Cy,Dx,Dy = coords
            self.x1 = min(Ax, min(Bx, min(Cx, Dx)))-1
            self.y1 = min(Ay, min(By, min(Cy, Dy)))-1
            self.x2 = max(Ax, max(Bx, max(Cx, Dx)))-1
            self.y2 = max(Ay, max(By, max(Cy, Dy)))-1
        self.cf = context_factor

    def set_xy(self, x1,x2,y1,y2):
        self.x1 = int(x1)
        self.x2 = int(x2)
        self.y1 = int(y1)
        self.y2 = int(y2)

    def get_center(self):
        return (self.x1+self.x2)/2, (self.y1+self.y2)/2

    def crop_frame(self, frame):
        frame = frame[self.y1:self.y2, self.x1:self.x2, :]
        for s in frame.shape:
            if s==0:
                print "crop_frame"
                import IPython
                IPython.embed()

        return frame

    def in_box(self, x,y):
        return x> self.x1 and x<self.x2 and y > self.y1 and y< self.y2

    def draw_frame(self, frame, channel):
        nframe = frame.copy()
        nframe[self.y1,self.x1:self.x2,channel] = 255
        nframe[self.y2-1,self.x1:self.x2,channel] = 255
        nframe[self.y1:self.y2,self.x1,channel] = 255
        nframe[self.y1:self.y2,self.x2-1,channel] = 255
        return nframe

    def compute_output_size(self):
        return  max(0,(self.y2-self.y1)*self.cf),max(0, (self.x2-self.x1)*self.cf)

    def compute_crop_loc(self, frame):
        center_x,center_y = self.get_center()
        im_h, im_w, _ = frame.shape
        out_h, out_w = self.compute_output_size()
        roi_left = max(0,center_x-out_w/2)
        roi_bottom = max(0, center_y-out_h/2)

        left_half = min(out_w/2, center_x)
        right_half = min(out_w/2, im_w-center_x)
        roi_width = max(1, left_half+right_half)
        top_half = min(out_h/2, center_y)
        bottom_half = min(out_h/2, im_h-center_y)
        roi_height = max(1.0, top_half+bottom_half)

        newBB = BBox(self.cf)
        newBB.set_xy(roi_left, roi_left+roi_width, roi_bottom, roi_bottom+roi_height)

        for s in [newBB.x2-newBB.x1, newBB.y2-newBB.y1]:
            if s == 0:
                print "compute_crop_loc"
                import IPython
                IPython.embed()

        return newBB

    def edge_spacing(self):
        out_h, out_w = self.compute_output_size()
        center_x,center_y = self.get_center()
        return max(0.0, out_w/2-center_x), max(0.0, out_h/2-center_y)

    def cropPadImage(self, frame):
        return self.get_next_crop(frame, False)

    def get_next_crop(self, frame, test):
        # print frame.shape
        if test:
            import IPython
            IPython.embed()
        im_h, im_w, im_c = frame.shape
        BB = self.compute_crop_loc(frame)
        roi_left = min(BB.x1, im_w-1.0)
        roi_bottom = min(BB.y1, im_h-1.0)
        roi_width = min(im_w*1.0, max(1.0, math.ceil(BB.x2-BB.x1)))
        roi_height = min(im_h*1.0, max(1.0, math.ceil(BB.y2-BB.y1)))
        BB.x1 = roi_left
        BB.y1 = roi_bottom
        BB.x2 = roi_left+roi_width
        BB.y2 = roi_bottom+roi_height
        cropped = BB.crop_frame(frame)
        out_h, out_w = self.compute_output_size()
        out_w = max(math.ceil(out_w), roi_width)
        out_h = max(math.ceil(out_h), roi_height)
        out_img = np.zeros((out_h, out_w, im_c))

        edge_x, edge_y = self.edge_spacing()
        edge_spacing_x = min(edge_x, float(out_w)-1)
        edge_spacing_y = min(edge_y, float(out_h)-1)
        # import IPython
        # IPython.embed()
        out_img[edge_spacing_y:(edge_spacing_y+roi_height), edge_spacing_x:(edge_spacing_x+roi_width), :] = cropped

        for s in out_img.shape:
            if s==0:
                import IPython
                IPython.embed()
        return out_img, BB, edge_x, edge_y #cropped, BB, edge_x, edge_y #

    def get_tight_bbox(self, points, next_crop, test, frame, edge_x, edge_y):
        # crop relative to next_crop
        if test:
            import IPython
            IPython.embed()
        # points[points<0] = 0.
        q1, q2 = points[[0,2]]*next_crop.shape[1]
        p1, p2 =  points[[1,3]]*next_crop.shape[0]

        real_x1 = max(q1+self.x1-edge_x,0)
        real_x2 = min(q2+self.x1-edge_x, frame.shape[1])
        real_y1 = max(p1+self.y1-edge_y,0)
        real_y2 = min(p2+self.y1-edge_y, frame.shape[0])
        tight_bbox = BBox(self.cf)

        for s in [real_x2-real_x1, real_y2-real_y1]:
            if s == 0:
                print "get_tight_bbox"
                import IPython
                IPython.embed()
        tight_bbox.set_xy(real_x1,real_x2, real_y1, real_y2)
        # import IPython; IPython.embed()

        return tight_bbox

    # def uncenter(self, image_curr, search_location, edge_x, edge_y):
