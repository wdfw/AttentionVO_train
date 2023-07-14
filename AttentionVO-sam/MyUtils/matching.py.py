from src.loftr import LoFTR, default_cfg
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor,process_resize)
import torch
import numpy as np
import cv2

class BASEMatching :
    def __init__(self,original_size, resize, device = 'cuda') : 
        
        self.resize = resize
        self.original_size = original_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.matching = None
        self.matcher = None
    def __call__(self, img1, img2) :
        ...
        
    def rescale(self, size0=None, size1=None) :
        size0 = self.resize
        size1 = self.original_size
            
        self.matching["mkpts0"][:,0] = self.matching["mkpts0"][:,0]*size1[0]//size0[0]
        self.matching["mkpts0"][:,1] = self.matching["mkpts0"][:,1]*size1[1]//size0[1]
                
        self.matching["mkpts1"][:,0] = self.matching["mkpts1"][:,0]*size1[0]//size0[0]
        self.matching["mkpts1"][:,1] = self.matching["mkpts1"][:,1]*size1[1]//size0[1]    


class SGMatching(BASEMatching) :
    def __init__(self,original_size,resize,device = 'cuda',conf_threshold = 0.9,config={}) :
        
        super().__init__(original_size,resize,device)

        self.matcher = Matching(config).eval().to(self.device)
        
        self.conf = conf_threshold
            
    def __call__(self, img0, img1) :
        
        if img0.shape[-1] == 3 : img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY )
        if img1.shape[-1] == 3 : img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY )
            
        w, h = img0.shape[1], img0.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
    

        img0 = cv2.resize(img0, (w_new, h_new)).astype('float32')
        img1 = cv2.resize(img1, (w_new, h_new)).astype('float32')
        with torch.no_grad():
            inp0 = frame2tensor(img0, self.device)
            inp1 = frame2tensor(img1, self.device)

            pred = self.matcher({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            valid = conf > self.conf        
            mconf = conf[valid]    
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]     
            self.matching = {"mkpts0" : mkpts0,"mkpts1" : mkpts1, "mconf" : mconf}
        
        return self.matching
    
class LoFTRMatching(BASEMatching) :       
    def __init__(self,original_size,resize,device = 'cuda',weight_dir=None,config={}) :
        super().__init__(original_size,resize,device)
        
        self.matcher = LoFTR(config=config)
        if weight_dir :
            self.matcher.load_state_dict(torch.load(weight_dir)['state_dict'])
            self.matcher = self.matcher.eval().cuda()
        self.resize = resize
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def __call__(self, img0, img1) :
       
        if img0.shape[-1] == 3 : img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY )
        if img1.shape[-1] == 3 : img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY )
        
        th,tw = self.resize
        img0 = cv2.resize(img0, (tw,th)).astype('float32')
        img1 = cv2.resize(img0, (tw,th)).astype('float32')

        
        img0 = torch.from_numpy(img0)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1)[None][None].cuda() / 255.
        
        batch = {'image0': img0, 'image1': img1}
        
        with torch.no_grad():
            self.matcher(batch)

            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()  
            self.matching = {"mkpts0" : mkpts0,"mkpts1" : mkpts1, "mconf" : mconf}
            
        return self.matching