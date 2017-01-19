import math
import numpy as np
import caffe
import cv2

class BBox:
    def __init__(self, context_factor, coords=None):
        if coords is not None:
            Ax,Ay,Bx,By,Cx,Cy,Dx,Dy = coords
            self.y1 = min(Ax, min(Bx, min(Cx, Dx)))-1
            self.x1 = min(Ay, min(By, min(Cy, Dy)))-1
            self.y2 = max(Ax, max(Bx, max(Cx, Dx)))-1
            self.x2 = max(Ay, max(By, max(Cy, Dy)))-1
        self.cf = context_factor

    def set_xy(self, x1,x2,y1,y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 =y2

    def get_center(self):
        return (self.x1+self.x2)/2, (self.y1+self.y2)/2

    def crop_frame(self, frame):
        frame = frame[self.x1:self.x2, self.y1:self.y2, :]
        return frame

    def draw_frame(self, frame):
        frame[self.x1,self.y1:self.y2,0] = 255
        frame[self.x2,self.y1:self.y2,0] = 255
        frame[self.x1:self.x2,self.y1,0] = 255
        frame[self.x1:self.x2,self.y2,0] = 255
        return frame

    def compute_output_size(self):
        return (self.x2-self.x1)*self.cf, (self.y2-self.y1)*self.cf

    def compute_crop_loc(self, frame):
        center_x,center_y = self.get_center()
        im_w, im_h, _ = frame.shape
        out_w, out_h = self.compute_output_size()
        roi_left = max(0,center_x-out_w/2)
        roi_bottom = max(0, center_y-out_h/2)

        left_half = min(out_w/2, center_x)
        right_half = min(out_w/2, im_w-center_x)
        roi_width = max(1, left_half+right_half)
        top_half = min(out_h/2, center_y)
        bottom_half = min(out_h/2, im_h-center_y)
        roi_height = max(1.0, top_half+bottom_half)

        newBB = BBox(self.cf)
        newBB.x1 = roi_left
        newBB.y1 = roi_bottom
        newBB.x2 = roi_left+roi_width
        newBB.y2 = roi_bottom+roi_height
        return newBB

    def edge_spacing(self):
        out_w, out_h = self.compute_output_size()
        center_x,center_y = self.get_center()
        return max(0.0, out_w/2-center_x), max(0.0, out_h/2-center_y)

    def get_next_crop(self, frame):
        print frame.shape
        im_w, im_h, im_c = frame.shape
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
        out_w, out_h = self.compute_output_size()
        out_w = max(math.ceil(out_w), roi_width)
        out_h = max(math.ceil(out_h), roi_height)
        out_img = np.zeros((out_w, out_h, im_c))

        edge_x, edge_y = self.edge_spacing()
        edge_spacing_x = min(edge_x, float(im_w)-1)
        edge_spacing_y = min(edge_y, float(im_h)-1)
        out_img[edge_spacing_x:roi_width, edge_spacing_y:roi_height, :] = cropped
        return out_img, BB

    def get_tight_bbox(self, points, next_crop):
        # crop relative to next_crop
        p1, p2 = points[[0,2]]*next_crop.shape[1]
        q1, q2 =  points[[1,3]]*next_crop.shape[0]

        real_x1 = q1+self.x1
        real_x2 = q2+self.x1
        real_y1 = p1+self.y1
        real_y2 = p2+self.y1
        tight_bbox = BBox(self.cf)
        tight_bbox.set_xy(real_x1,real_x2, real_y1, real_y2)
        return tight_bbox
