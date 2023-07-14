import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from .matching import *
from collections import Counter
def BBOX(label,mkpt,img,threshold = 0.3) :
    total = label.size
    
    counter = Counter(label)
    common_label,common_size = counter.most_common(1)[0]
    
    if common_size / total > threshold :
        
        common_point = mkpt[label == common_label]
        up,but,left,right = np.min(common_point[:,1]),np.max(common_point[:,1]),np.min(common_point[:,0]),np.max(common_point[:,0])
        
        #plt.scatter(mkpt[:,0],mkpt[:,1])
        #plt.xlim([left,right])
        #plt.ylim([but,up])        
        #plt.imshow(img)
        #plt.show()
        return np.array([up,but,left,right],dtype = np.uint16)
    else : 
        h,w = img.shape[:-1]
        return np.array([0,h,0,w],dtype = np.uint16)
        
class Matching_and_Crop :
    def __init__(self, matchings,threshold = 0.3,eps = 40, min_samples=10) :
        self.min_samples = min_samples
        self.eps = eps
        self.threshold = threshold
        self.sequence = matchings[:]
    def __call__(self,sample) :
    
        img1 = sample["img1"]
        img2 = sample["img2"]
        flow = sample["flow"]
        
        concate_mkpt = {"mkpts0" : [], 
                    "mkpts1" : []}
        for matcher in self.sequence :
            matcher(img1,img2)
            matcher.rescale()
            concate_mkpt["mkpts0"].append(matcher.matching['mkpts0'])
            concate_mkpt["mkpts1"].append(matcher.matching['mkpts1'])
            
        concate_mkpt = {"mkpts0" : np.concatenate([*concate_mkpt["mkpts0"]]), 
                        "mkpts1" : np.concatenate([*concate_mkpt["mkpts1"]])}   
        concate_mkpt = {"mkpts0" : np.unique(concate_mkpt["mkpts0"],axis = 0), 
                        "mkpts1" : np.unique(concate_mkpt["mkpts1"],axis = 0) }
                        
        if len(concate_mkpt["mkpts0"]) :
            clustering=DBSCAN(eps=self.eps,min_samples=self.min_samples).fit(concate_mkpt["mkpts0"]) 
        
            area = BBOX(clustering.labels_,concate_mkpt["mkpts0"],img1,threshold=self.threshold)

            crop_img1 = np.zeros(img1.shape,dtype=np.uint8)
            crop_img1[area[0]:area[1],area[2]:area[3]] = img1[area[0]:area[1],area[2]:area[3]]

            crop_img2 = np.zeros(img2.shape,dtype=np.uint8)
            crop_img2[area[0]:area[1],area[2]:area[3]] = img2[area[0]:area[1],area[2]:area[3]]

            crop_flow = np.zeros(flow.shape,dtype=np.uint8)
            crop_flow[area[0]:area[1],area[2]:area[3]] = flow[area[0]:area[1],area[2]:area[3]]
        else :
            crop_img1 = img1
            crop_img2 = img2
            crop_flow = flow
        return {"img1":img1, "img2":img2,"flow":flow, "crop_img1" : crop_img1,"crop_img2" : crop_img2,"crop_flow":crop_flow}
