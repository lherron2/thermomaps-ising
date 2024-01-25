import os
import numpy as np
# import matplotlib.pyplot as plt

from slurmflow.serializer import ObjectSerializer
from data.generic import Summary

import argparse

import logging
# logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tm.core.DiffusionModel').setLevel(logging.INFO)
logging.getLogger('tm.core.DiffusionProcesses').setLevel(logging.INFO)
logging.getLogger('tm.core.Loader').setLevel(logging.INFO)
logging.getLogger('tm.core.Prior').setLevel(logging.INFO)

class TrainingTracker:
    def __init__(self, trainer, summary):
        self.test_losses = trainer.test_losses
        self.train_losses = trainer.train_losses
        self.summary = summary

def train(args):

    from tm.core.backbone import ConvBackbone
    from tm.core.diffusion_model import DiffusionTrainer
    from tm.core.diffusion_process import VPDiffusion
    from tm.core.prior import GlobalEquilibriumHarmonicPrior, UnitNormalPrior
    from tm.architectures.unet_2d_mid_attn import Unet2D as Unet2D

    # Initialize the dataset
    OS = ObjectSerializer(args.train_loader_path)
    train_loader = OS.load()
    OS = ObjectSerializer(args.test_loader_path)
    test_loader = OS.load()

    # Initialize the prior
    if args.prior == "GEHP":
        prior = GlobalEquilibriumHarmonicPrior(shape=train_loader.data.shape, 
                                            channels_info={"coordinate": [0], "fluctuation": [1]})
    elif args.prior == "UNP":
        prior = UnitNormalPrior(shape=train_loader.data.shape, 
                                channels_info={"coordinate": [0], "fluctuation": [1]})

    # Initialize the model
    model = Unet2D(dim=args.dim, dim_mults=(1, 2, 4), resnet_block_groups=8, channels=2)
    backbone = ConvBackbone(model=model, data_shape=train_loader.data_dim, target_shape=8, num_dims=4,
                            lr=args.learning_rate, eval_mode="train", interpolate=False, self_condition=False)
    diffusion = VPDiffusion(num_diffusion_timesteps=100)
    trainer = DiffusionTrainer(diffusion, backbone, train_loader, model_dir=args.model_dir, pred_type=args.pred_mode, 
                            prior=prior, device='cuda:0', test_loader=test_loader)

    # training for 20 epochs
    trainer.train(10, loss_type="smooth_l1", batch_size=args.batch_size, print_freq=100)

    # Recording relevant information from the training run
    summary = Summary(**{k:v for k,v in args.__dict__.items() if '__' not in k})
    tracker = TrainingTracker(trainer, summary)
    OS = ObjectSerializer(os.path.join(args.model_dir, "tracker.h5"))
    OS.serialize(tracker)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Thermomaps.')
    
    # Create subparsers for the two modes
    subparsers = parser.add_subparsers(dest='mode')

    # Single run parser
    train_parser = subparsers.add_parser('train', help='Run a single simulation')

    # Add arguments
    train_parser.add_argument('--prior', type=str, default="GEHP", help='Prior type')
    train_parser.add_argument('--sampler', type=str, default="ClusterFlip", help='Sampler type')
    train_parser.add_argument('--size', type=int, default=8, help='Size parameter')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--dim', type=int, default=8, help='Dimension')
    train_parser.add_argument('--pred_mode', type=str, default="noise", help='Prediction mode')
    train_parser.add_argument('--train_loader_path', type=str, default=None, help='Config file')
    train_parser.add_argument('--test_loader_path', type=str, default=None, help='Config file')
    train_parser.add_argument('--model_dir', type=str, default=None, help='Config file')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise ValueError("Invalid mode selected. Choose 'train'.")

if __name__ == "__main__":
    main()



