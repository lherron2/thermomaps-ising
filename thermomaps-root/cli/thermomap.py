import os
import torch
import argparse
import numpy as np
from tm.core.backbone import ConvBackbone
from tm.core.diffusion_model import DiffusionTrainer, SteeredDiffusionSampler
from tm.core.diffusion_process import VPDiffusion
from tm.core.loader import Loader
from tm.core.prior import LocalEquilibriumHarmonicPrior, GlobalEquilibriumHarmonicPrior
from tm.architectures.unet_2D_mid_attn import Unet2D
from tm.core.utils import compute_model_dim
from cli.cli_utils import populate_args_from_config, save_dict_as_npy, load_diffusion_data


def init_directory(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_devices = torch.cuda.device_count()

    return Directory(
        args.pdb,
        args.expid,
        args.iter,
        args.identifier,
        device,
        num_devices,
        experiment_path=args.experiment_path,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        temperature_path=args.temperature_path,
        fluctuation_path=args.fluctuation_path,
        sample_path=args.sample_path,
    )


def init_backbone(directory, loader, args):
    model_dim = compute_model_dim(loader.data_dim, groups=8)
    model = Unet2D(
        dim=model_dim,
        dim_mults=(1, 2, 2, 4),
        resnet_block_groups=8,
        learned_variance=False,
        self_condition=eval(args.self_condition),
        learned_sinusoidal_cond=True,
        channels=5,
    )

    return ConvBackbone(
        model=model,
        data_shape=loader.data_dim,
        target_shape=model_dim,
        num_dims=4,
        lr=1e-3,
        eval_mode="train",
        self_condition=eval(args.self_condition),
    )


def train(args):
    # Code specific to training
    directory = init_directory(args)
    rvecs = np.load(directory.fluct_path, allow_pickle=True).item(0)
    temps = np.array(list(rvecs.keys()))
    if args.fluctuation_mapping:
        prior = LocalEquilibriumHarmonicPrior(
            data=rvecs, temps=temps, cutoff=args.fluctuation_cutoff, fit_key="linear"
        )
    else:
        prior = GlobalEquilibriumHarmonicPrior(
            data=rvecs, temps=temps, cutoff=args.fluctuation_cutoff, fit_key="linear"
        )

    loader = Loader(
        directory,
        control_tuple=([1], [4, 5]),
        transform_type="identity",
        dequantize=True,
        dequantize_scale=1e-2,
    )

    backbone = init_backbone(directory, loader, args)
    diffusion = VPDiffusion(num_diffusion_timesteps=args.diffusion_timesteps)

    trainer = DiffusionTrainer(
        diffusion, backbone, loader, directory, args.pred_type, prior
    )
    trainer.train(
        args.num_epochs, loss_type="l2", batch_size=args.batch_size, print_freq=100
    )


def sample(args):
    directory = init_directory(args)
    # Code specific to sampling
    rvecs = np.load(directory.fluct_path, allow_pickle=True).item(0)
    temps = np.array(list(rvecs.keys()))
    prior = LocalEquilibriumHarmonicPrior(
        data=rvecs, temps=temps, fit_key="linear", cutoff=args.fluctuation_cutoff
    )

    loader = Loader(
        directory,
        control_tuple=([1], [4, 5]),
        transform_type="identity",
        dequantize_scale=1e-2,
    )

    backbone = init_backbone(directory, loader, args)
    diffusion = VPDiffusion(num_diffusion_timesteps=args.diffusion_timesteps)

    backbone.load_model(directory, args.epoch)

    sampler = SteeredDiffusionSampler(
        diffusion, backbone, loader, directory, args.pred_type, prior, gamma=1
    )
    sampler.sample_loop(
        args.num_samples, args.batch_size, args.pdb, int(args.gen_temp), n_ch=5
    )


def postprocess_samples(args):
    diffusion_dict = load_diffusion_data(args.diffusion_path, args.subsample)
    diffusion_file = os.path.join(args.data_path, args.outfile)
    save_dict_as_npy(diffusion_file, diffusion_dict)


def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Operations")
    parser.add_argument("--config_file", type=str, help="Path to a YAML config file.")

    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run.")

    # Train subparser
    parser_train = subparsers.add_parser("train", help="Train a diffusion model.")
    parser_train.add_argument(
        "--pdb", required=True, type=str, help="Path to the PDB file."
    )
    parser_train.add_argument(
        "--iter", required=True, type=str, help="Number of iterations."
    )
    parser_train.add_argument("--expid", required=True, type=str, help="Experiment ID.")
    parser_train.add_argument(
        "--pred_type", required=True, type=str, help="Prediction type."
    )
    parser_train.add_argument(
        "--num_epochs",
        type=int,
        default=250,
        help="Number of training epochs. Default is 250.",
    )
    parser_train.add_argument(
        "--self_condition",
        type=str,
        default="True",
        help="Whether to self-condition. Default is True.",
    )
    parser_train.add_argument(
        "--diffusion_timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps. Default is 100.",
    )
    parser_train.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training. Default is 16.",
    )
    parser_train.add_argument(
        "--fluctuation_cutoff",
        type=float,
        default=5e-2,
        help="Fluctuation cutoff value. Default is 5e-2.",
    )
    parser_train.add_argument(
        "--experiment_path",
        required=True,
        type=str,
        help="Path to the experiment directory.",
    )
    parser_train.add_argument(
        "--model_path", required=True, type=str, help="Path to the model directory."
    )
    parser_train.add_argument(
        "--dataset_path", required=True, type=str, help="Path to the dataset directory."
    )
    parser_train.add_argument(
        "--temperature_path",
        required=True,
        type=str,
        help="Path to the temperature data.",
    )
    parser_train.add_argument(
        "--fluctuation_path",
        required=True,
        type=str,
        help="Path to the fluctuation data.",
    )
    parser_train.add_argument(
        "--sample_path", required=True, type=str, help="Path to the sample data."
    )
    parser_train.add_argument(
        "--identifier",
        required=True,
        type=str,
        help="Identifier for the experiment/model.",
    )
    parser_train.add_argument(
        "--fluctuation_mapping",
        action="store_true",
        help="Flag to use LocalEquilibriumHarmonicPrior. If not provided, GlobalEquilibriumHarmonicPrior will be used.",
    )

    # Sample subparser
    parser_sample = subparsers.add_parser(
        "sample", help="Sample using a diffusion model."
    )
    parser_sample.add_argument(
        "--pdb", required=True, type=str, help="Path to the PDB file."
    )
    parser_sample.add_argument(
        "--iter", required=True, type=str, help="Number of iterations."
    )
    parser_sample.add_argument(
        "--expid", required=True, type=str, help="Experiment ID."
    )
    parser_sample.add_argument("--epoch", required=True, type=int, help="Epoch number.")
    parser_sample.add_argument(
        "--gen_temp", required=True, type=float, help="Generation temperature."
    )
    parser_sample.add_argument(
        "--pred_type", required=True, type=str, help="Prediction type."
    )
    parser_sample.add_argument(
        "--num_samples", required=True, type=int, help="Number of samples to generate."
    )
    parser_sample.add_argument(
        "--self_condition",
        type=str,
        default="True",
        help="Whether to self-condition. Default is True.",
    )
    parser_sample.add_argument(
        "--diffusion_timesteps",
        type=int,
        default=100,
        help="Number of diffusion timesteps. Default is 100.",
    )
    parser_sample.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Batch size for sampling. Default is 5000.",
    )
    parser_sample.add_argument(
        "--fluctuation_cutoff",
        type=float,
        default=5e-2,
        help="Fluctuation cutoff value. Default is 5e-2.",
    )
    parser_sample.add_argument(
        "--experiment_path",
        required=True,
        type=str,
        help="Path to the experiment directory.",
    )
    parser_sample.add_argument(
        "--model_path", required=True, type=str, help="Path to the model directory."
    )
    parser_sample.add_argument(
        "--dataset_path", required=True, type=str, help="Path to the dataset directory."
    )
    parser_sample.add_argument(
        "--temperature_path",
        required=True,
        type=str,
        help="Path to the temperature data.",
    )
    parser_sample.add_argument(
        "--fluctuation_path",
        required=True,
        type=str,
        help="Path to the fluctuation data.",
    )
    parser_sample.add_argument(
        "--sample_path", required=True, type=str, help="Path to the sample data."
    )
    parser_sample.add_argument(
        "--identifier",
        required=True,
        type=str,
        help="Identifier for the experiment/model.",
    )

    # Postprocess samples subparser
    parser_postprocess = subparsers.add_parser(
        "postprocess_samples", help="Postprocess samples from diffusion model."
    )
    parser_postprocess.add_argument("--pdbid", required=True, type=str, help="PDB ID.")
    parser_postprocess.add_argument(
        "--diffusion_path", type=str, help="Path to the diffusion data."
    )
    parser_postprocess.add_argument(
        "--subsample", type=int, help="Factor by which to subsample the data."
    )
    parser_postprocess.add_argument(
        "--data_path", type=str, help="Path to save the processed data."
    )
    parser_postprocess.add_argument(
        "--outfile",
        type=str,
        help="Name of the output file to save the processed data.",
    )

    args = parser.parse_args()

    subparsers_dict = {
        "train": parser_train,
        "sample": parser_sample,
        "postprocess_samples": parser_postprocess,
    }

    if args.config:
        args = populate_args_from_config(args, parser, subparsers_dict)

    # If a sub-command is provided, call the relevant function
    if args.command:
        args.func(args)
    else:
        parser.print_help()


