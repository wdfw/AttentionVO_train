from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset, TrajFolderDataset_flow
from Datasets.transformation import ses2poses_quat, tartan2kitti
from Datasets.tartanair import *
from Datasets.utils import *
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO


import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    testvo = TartanVO(args.model_name)

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    else:
        datastr = 'tartanair'
        
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)
    normalize = SampleNormalize(flow_norm=20) 
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), normalize,ToTensor()])
    testDataset = TrajFolderDataset_flow(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)

    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    motionlist = []
    testname = datastr + '_' + args.model_name.split('/')[-1].split('.')[0]

    while True:
        try:
            sample = next(testDataiter)
        except StopIteration:
            break

        motions = testvo.test_batch_onlypose(sample)
        motionlist.extend(motions)

    poselist = ses2poses_quat(np.array(motionlist))
    
    root="/drone/AttentionVO/"
    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        print("datastr=",datastr)
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname=root+'results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt(root+'results/'+testname+'.txt',results['est_aligned'])
    else:
        np.savetxt(root+'results/'+testname+'.txt',poselist)
    np.savetxt(root+'results/'+testname+'_gt-kitti_format.txt',tartan2kitti(np.loadtxt(args.pose_file)))
    np.savetxt(root+'results/'+testname+'-kitti_format.txt',tartan2kitti(poselist))