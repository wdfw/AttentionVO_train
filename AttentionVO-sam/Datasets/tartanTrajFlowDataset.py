import numpy as np
import cv2
import re
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        # GT when inference for calc scale to align
        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.rgbfiles)) - 1
        else:
            self.motions = None

        self.N = len(self.rgbfiles) - 1

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        res = {'img1': img1, 'img2': img2 }

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is None:
            return res
        else:
            # GT when inference for calc scale to align
            res['motion'] = self.motions[idx]
            return res

class TrajFolderDataset_flow(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, flowfolder , posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        files = listdir(flowfolder)
        rx_flow = re.compile(r"\.npy$", re.MULTILINE | re.IGNORECASE)
        self.flowfiles = [f"{flowfolder}/{f}"
                            for f in files if rx_flow.search(f)]
        self.flowfiles.sort()
        # self.rgbfiles = [(flowfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        # self.rgbfiles.sort()
        self.flowfolder = flowfolder

        print('Find {} flow files in {}'.format(len(self.flowfiles), flowfolder))

        # GT when inference for calc scale to align
        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            print(len(self.motions), len(self.flowfiles))
            assert(len(self.motions) == len(self.flowfiles)+1) -1
        else:
            self.motions = None

        self.N = len(self.flowfiles)

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        flowFile = self.flowfiles[idx]
        res = {"flow":np.load(flowFile)}

        

        h, w, _ = 480,640,None
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is None:
            return res
        else:
            # GT when inference for calc scale to align
            res['motion'] = self.motions[idx]
            return res
    # def __getitem__(self, idx):
    # #用來輸入pwcnet的光流
    #     flowFile = self.flowfiles[idx]
    #     # res = {"flow":np.load(flowFile).transpose(2,0,1)}
    #     res = {"flow":np.load(flowFile)}

    #     h, w, _ = 480,640,None
    #     intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
    #     res['intrinsic'] = intrinsicLayer
    #     print()
    #     print(res["flow"].shape)

    #     if self.transform:
    #         res = self.transform(res)
    #     print(res["flow"].shape)

    #     if self.motions is None:
    #         return res
    #     else:
    #         # GT when inference for calc scale to align
    #         res['motion'] = self.motions[idx]
    #         return res
