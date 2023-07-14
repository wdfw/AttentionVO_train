
import numpy as np
import cv2
class Attention(object):

    def __init__(self, resize, matcher):       
        self.resize = resize
        self.matcher = matcher
        
    def __call__(self, sample):

        th, tw = self.resize
        h, w = sample['img1'].shape[0], sample['img1'].shape[1]
    
        #if w == tw and h == th:
        #    return sample
        
        intrinsic = sample['intrinsic']
        
        crop_result = self.matcher(sample)
        
        img1 = cv2.resize(crop_result["crop_img1"], (tw,th), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(crop_result["crop_img2"], (tw,th), interpolation=cv2.INTER_LINEAR)
        flow =  cv2.resize(crop_result["crop_flow"], (tw,th), interpolation=cv2.INTER_LINEAR)
        intrinsic = cv2.resize(intrinsic, (tw,th), interpolation=cv2.INTER_LINEAR)  
        
        sample = {'img1':img1,'img2':img2, 'intrinsic':intrinsic,'flow':flow,'motion':sample['motion']}
        return sample