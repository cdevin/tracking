import argparse
import rospy
import roslib
from sensor_msgs.msg import Image
# from ddp_controller_pkg.msg import ImgFeatures
import numpy as np
import time

class Image_Processor():
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        # self._network = network
        # self._event = event
        # self.publisher = rospy.Publisher(self.hyperparams["publish_topic"], ImgFeatures)
        self.images = []
        self.num = 0

    def process_images(self):
        rospy.init_node('img_processor', anonymous=True)
        rospy.Subscriber(self._hyperparams["subscribe_topic"], Image, self.process)
        self.time = time.clock()
        rospy.spin()

    def process(self, image):
    	# if self._event.isSet():
      	#  rospy.signal_shutdown()

        image_data = np.fromstring(image.data, np.uint8).reshape(image.height, image.width, 3)[::-1, :, ::-1]
        # image_data = image_data[self._hyperparams["vertical_crop"]:image.height - self._hyperparams["vertical_crop"],
    	#                         self._hyperparams["horizontal_crop"]:image.height - self._hyperparams["horizontal_crop"]]

        # Put data through the nueral network
        # self._network.blobs[self.net.blobs.keys()[0]].data[:] = image_data
        # features = self._network.forward().values()[0][0]
        # features = self._network(image_data)
        # self.publisher.publish(features)
        self.images.append(image_data)
        if len(self.images) == 200:
            # import IPython
            # IPython.embed()
            images = np.array(self.images)
            time2 = time.clock()
            print "fps", len(self.images)/(time2-self.time)
            np.save(self._hyperparams['file_prefix']+str(self.num)+'.npy', images)
            print "saved", self.num
            self.num +=1
            self.images = []
            self.time = time.clock()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('topic', type=str)
    parser.add_argument('saveto', type=str)
    args = parser.parse_args()
    hyperparams = {'subscribe_topic': args.topic, 'file_prefix': args.saveto}
    ip = Image_Processor(hyperparams)
    ip.process_images()
