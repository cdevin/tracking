import numpy as np
from tracker import Tracker

class Tracker_npy(Tracker):
    def __init__(self, context_factor=2):
        Tracker.__init__(self, context_factor)

    def load_full_video(self, file_prefix):
        self.current_dir = file_prefix
        self.current_frame = 0
        self.current_file =0
        next_frames = np.load(self.current_dir+str(self.current_file)+'.npy')
        frames = []
        while next_frames is not None:
            frames += [next_frames[f] for f in range(next_frames.shape[0])]
            self.current_file += 1
            try: 
                next_frames = np.load(self.current_dir+str(self.current_file)+'.npy')
            except IOError:
                next_frames = None
        self.all_frames = frames

        
