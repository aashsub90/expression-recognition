'''
This library will
1. Capture a image from webcam and store it to disk. (capture_interval, storage_location)
2. Read the image from stored location.
'''

import cv2
import random
import os

class image_processor:
    def __init__(self, path=None):
        self.store = path
        #self.freq = frequency

    def create_folder(self):
        current_loc = os.getcwd()
        path = current_loc+'/'+self.store
        os.mkdir(path, 0o755)

    def image_capture(self):
        camera = cv2.VideoCapture(0)
        for i in range(10):
            retun_value, image = camera.read()
            # self.create_folder()
            cv2.imwrite(self.store+'/opencv_'+str(random.randint(10,20))+'.png',image)
        del(camera)


    def image_read(self):
        loc = self.store
        print(loc)
        pass


# loc : must be the path of an existing folder.
loc = ''
ip = image_processor(loc)
ip.image_capture()




