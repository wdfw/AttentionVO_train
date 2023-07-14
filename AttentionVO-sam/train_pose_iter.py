import argparse
import copy
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from evo.core import metrics as evo_metric
from evo.core.trajectory import PosePath3D
from ruamel.yaml import YAML
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

import hydra
import random

from Datasets.tartanair import *
from Datasets.transformation import ses2poses_quat
from Datasets.utils import *
from Network.loss import *
from Network.utils import *
from TartanVO import TartanVO

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
# MODEL_DIR = ROOT / "models"
MODEL_DIR = Path("/media/sam/本機磁碟/tartanvo/model")
LOG_DIR = ROOT / "logs"
# DATASET_DIR = Path("/drone/tartanair_tools/dataset/data")
DATASET_DIR = Path("/media/sam/4TSSD1/Dataset/tartanair")
CONFIC_FILE = ROOT / "configs" / "train.yaml"
CKPNT_DIR = MODEL_DIR / "checkpoints"
MODEL_FNAME = "tartanvo_{0:%Y%m%d_%H%M}_{1:}.pkl"


def get_args(cfg):
    parser = argparse.ArgumentParser(description="HRL")

    parser.add_argument(
        "--image-width", type=int, default=640, help="image width (default: 640)"
    )
    parser.add_argument(
        "--image-height", type=int, default=448, help="image height (default: 448)"
    )
    parser.add_argument(
        "--model-name", default=str(cfg.model.name), help='name of pretrained model (default: "")'
    )
    parser.add_argument(
        "--euroc",
        action="store_true",
        default=False,
        help="euroc test (default: False)",
    )
    parser.add_argument(
        "--kitti",
        action="store_true",
        default=False,
        help="kitti test (default: False)",
    )
    parser.add_argument(
        "--kitti-intrinsics-file",
        default="",
        help="kitti intrinsics file calib.txt (default: )",
    )
    parser.add_argument(
        "--test-dir",
        default="",
        help='test trajectory folder where the RGB images are (default: "")',
    )
    parser.add_argument(
        "--flow-dir",
        default="",
        help='test trajectory folder where the optical flow are (default: "")',
    )
    parser.add_argument(
        "--pose-file",
        default="",
        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")',
    )
    parser.add_argument(
        "--save-flow",
        action="store_true",
        default=False,
        help="save optical flow (default: False)",
    )

    args = parser.parse_args()

    return args


class Trainer():
    def __init__(self, cfg, args):
        self.args = args
        self.cfg = cfg
        # load trajectory data from a folder
        datastr = "tartanair"
        if args.kitti:
            datastr = "kitti"
        elif args.euroc:
            datastr = "euroc"
        else:
            datastr = "tartanair"
        fx, fy, cx, cy = dataset_intrinsics(datastr)
        if args.kitti_intrinsics_file.endswith(".txt") and datastr == "kitti":
            fx, fy, cx, cy = load_kiiti_intrinsics(
                args.kitti_intrinsics_file
            )
            # pose normalize args is omitted
        self.normalize = SampleNormalize(flow_norm=20)
        self.transform = Compose([
            CropCenter((args.image_height, args.image_width)),
            DownscaleFlow(),
            self.normalize,
            ToTensor(),
        ])

        # Load configs
        yaml = YAML(typ='safe')
        with open("/drone/AttentionVO/configs/train_onlypose.yaml", "r") as stream:
            self.config = yaml.load(stream)
        # model
        tartanvo = TartanVO(args.model_name)
        self.model = tartanvo.vonet
        # freeze_params(self.model.flowNet) # fix flow net

        # Dataset
        intrinsics = CameraIntrinsics(fx, fy, cx, cy)
        # tartanair_set = TartanAirDataset(f"{DATASET_DIR / 'TartanAir'}",
        tartanair_set = TartanAirDataset(DATASET_DIR,
                                         intrinsics, transform=self.transform)

        tartanair_set.environment=tartanair_set.train_seq_name
        self.train_set = ConcatDataset(tartanair_set.get_list)

        tartanair_set.environment=tartanair_set.val_seq_name
        self.valid_set = ConcatDataset(tartanair_set.get_list)

        # all_seq_set = ConcatDataset(tartanair_set.all)
        # train_size = int(0.8 * len(all_seq_set))
        # test_size = len(all_seq_set) - train_size
        # self.train_set, self.valid_set = torch.utils.data.random_split(all_seq_set,
        #                                                      [train_size,
        #                                                          test_size],
        #                                                      torch.Generator().manual_seed(42))    #

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Hyper parameters
        self.hparams = self.config["hparams"]
        print(self.hparams)
        
        optimizer = getattr(torch.optim, self.hparams["optimizer"])
        self.optimizer = optimizer(
            params=self.model.parameters(), lr=self.hparams["lr"])

        num_workers = 0
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.hparams["batch_size"],
                                       shuffle=True,
                                       num_workers=num_workers)
        # Shuffle -> use RPE metric
        self.valid_loader = DataLoader(self.valid_set,
                                       batch_size=self.hparams["batch_size"],
                                       shuffle=True,
                                       num_workers=num_workers)
        print(len(self.train_loader))
        print(len(self.valid_loader))
        input()

        # Loss
        self.flow_criterion = FlowLoss()
        self.pose_criterion = PoseNormLoss(1e-6)
        self.total_criterion = WeightedLoss([self.hparams["lambda"], 1])

        # Metric (EVO settings)
        pose_relation = evo_metric.PoseRelation.full_transformation
        rpe_metric = getattr(evo_metric, self.config["metric_type"])
        self.rpe_metric = rpe_metric(pose_relation=pose_relation, delta=1,
                                     delta_unit=evo_metric.Unit.frames, all_pairs=False)

        # Logger
        log_folder = LOG_DIR / \
            f"{datetime.now():%Y%m%d_%H%M}_{self.cfg.log_name}"
        self.writer = SummaryWriter(f"{log_folder}")

        self.model.train()
        self.best_metirc = math.inf
        self.train_loss = {k: 0 for k in ["pose", "total"]}
        self.valid_loss = copy.deepcopy(self.train_loss)
        self.batch_loss = copy.deepcopy(self.train_loss)
        self.metric = {k: 0 for k in self.config["metrics"]}

        self.keep_training = True
        self.iter = 0

    def train(self):
        while(self.keep_training):
            self.Train_one_epoch()
            self.Validation()

    def Train_one_epoch(self):
        # Training
        # Clear loss
        # for k in train_loss:
        #     train_loss[k] = 0

        for i, sample in enumerate(tqdm(self.train_loader, desc="#Train batch", leave=False)):
            # Data
            # img0 = sample["img1"].to(device)
            # img1 = sample["img2"].to(device)
            intrinsic = sample["intrinsic"].to(self.device)
            flow = sample["flow"].to(self.device)  # N x C x H x W

            # Ground truth
            pose_gt = sample["motion"].to(self.device)  # N x 6

            # Forward
            pose = self.model([flow, intrinsic])

            # Batch loss
            # Pre-divide by batch size to make it numerically stable
            # batch_loss["flow"] = flow_criterion(flow, flow_gt).sum()
            self.batch_loss["pose"] = self.pose_criterion(pose, pose_gt).sum()
            self.batch_loss["total"] = self.batch_loss["pose"]
            # Instant preview loss
            tmp = {k: (v / self.train_loader.batch_size).item()
                   for k, v in self.batch_loss.items()}
            tqdm.write(f"Train step {self.iter:<10}: {tmp}")

            # ep
            # Not accurate (a little bit) if all batch size (=denominator) are not the same
            # train_loss["flow"] += batch_loss["flow"].item()
            # train_loss["pose"] += batch_loss["pose"].item()
            # train_loss["total"] += batch_loss["total"].item()
            # Backpropagation
            self.optimizer.zero_grad()
            self.batch_loss["total"].backward()
            self.optimizer.step()

            # Log train
            for k in tmp:
                self.writer.add_scalars(
                    k, {"train": tmp[k]}, self.iter)

            self.iter += 1
            if self.iter >= self.hparams["iteration"]:
                self.keep_training = False
                return

            #1e-4 with a decay rate of 0.2 at 1/2 and 7/8 of the total trainingstep
            elif self.iter == 50 : #75000
                for g in self.optimizer.param_groups:
                    g['lr'] = self.hparams["lr"] * 0.2
        return
        # loss / train_set.size
        # step_num = len(train_set) if keep_training else (iter % len(train_set))
        # for k, v in train_loss.items():
        #     train_loss[k] = v / step_num
        # tqdm.write(f"Epoch {ep}: Train: loss= {train_loss}")

    def Validation(self):
        # Validation
        with torch.no_grad():
            # Clear loss (metric just overwrite)
            for k in self.valid_loss:
                self.valid_loss[k] = 0
            pose_list = []
            pose_gt_list = []
            flow_img_list = []
            self.model.eval()
            for i, sample in enumerate(tqdm(self.valid_loader, desc="#Valid batch", leave=False)):
                # Data
                # img0 = sample["img1"].to(device)
                # img1 = sample["img2"].to(device)
                intrinsic = sample["intrinsic"].to(self.device)
                flow = sample["flow"].to(self.device)  # N x C x H x W

                # Ground truth
                pose_gt = sample["motion"].to(self.device)  # N x 6

                # Forward
                pose = self.model([flow, intrinsic])

                # Batch loss
                # Pre-divide by batch size to make it numerically stable
                # batch_loss["flow"] = flow_criterion(flow, flow_gt).sum()
                self.batch_loss["pose"] = self.pose_criterion(
                    pose, pose_gt).sum()
                self.batch_loss["total"] = self.batch_loss["pose"]

                # Total loss
                # Not accurate (a little bit) if all batch size (=denominator) are not the same
                # valid_loss["flow"] += batch_loss["flow"].item()
                self.valid_loss["pose"] += self.batch_loss["pose"].item()
                self.valid_loss["total"] += self.batch_loss["total"].item()

                # Denormalize to evaluate by metric
                # No need to detach() under no_grad() context

                _, pose_np = self.normalize.denormalize(
                    pose=pose.cpu().numpy())
                _, pose_gt_np = self.normalize.denormalize(
                    pose=pose_gt.cpu().numpy())

                # Gather output to evaluate
                pose_list.extend(pose_np)
                pose_gt_list.extend(pose_gt_np)

                if self.args.save_flow:
                    # B*C*H*W -> B*H*W*C
                    flow_np = flow.permute(0, 2, 3, 1).cpu().numpy()
                    flow_np = [visflow(flow_np[b])
                               for b in range(flow_np.shape[0])]
                    flow_img_list.extend(flow_np)
            # loss / valid_set.size
            for k, v in self.valid_loss.items():
                self.valid_loss[k] = v / len(self.valid_set)

            # Compute metric
            # Convert from se3 to t & q
            pose_list = ses2poses_quat(np.array(pose_list))
            pose_gt_list = ses2poses_quat(np.array(pose_gt_list))
            # Conver to evo type
            pose_list = PosePath3D(positions_xyz=pose_list[:, :3],
                                   orientations_quat_wxyz=pose_list[:, 3:])
            pose_gt_list = PosePath3D(positions_xyz=pose_gt_list[:, :3],
                                      orientations_quat_wxyz=pose_gt_list[:, 3:])
            # Align trajactory
            pose_list.align(pose_gt_list, correct_scale=True,
                            correct_only_scale=False)
            # Compute
            self.rpe_metric.process_data((pose_gt_list, pose_list))

            # Log train & valid loss
            for k in self.valid_loss:
                self.writer.add_scalars(
                    k, {"valid": self.valid_loss[k]}, self.iter)

            # Log metric
            # Pick required metrics only
            for k, v in self.rpe_metric.get_all_statistics().items():
                if k in self.config["metrics"]:
                    self.metric[k] = v
            self.writer.add_scalars(
                self.config["metric_type"], self.metric, self.iter)
            tqdm.write(
                f"Step {self.iter}: Valid: loss= {self.train_loss}, metric= {self.metric}")

            # Log flow image
            if self.args.save_flow:
                self.writer.add_image(f"Valid flows", np.array(
                    flow_img_list), self.iter, dataformats='NHWC')

            # TODO: handle overfitting
            # Model checkpoint. Select based on RPE if no RNN, else ATE
            if self.metric[self.config["ckpnt_metric"]] < self.best_metirc:
                if not os.path.isdir(MODEL_DIR / self.cfg.log_name):
                    os.mkdir(MODEL_DIR / self.cfg.log_name)
                best_metirc = self.metric[self.config["ckpnt_metric"]]
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, MODEL_DIR / self.cfg.log_name / MODEL_FNAME.format(datetime.now(), self.iter))
        return


@hydra.main(config_path='configs', config_name='train_onlypose')
def main(cfg):
    args = get_args(cfg)

    # set randomSeeds 42
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(cfg, args)
    trainer.train()
    print("\nFinished!")


if __name__ == "__main__":
    main()
