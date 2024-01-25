import numpy as np
import os
from slurmflow.config import ConfigParser
from slurmflow.serializer import ObjectSerializer
from slurmflow.driver import SlurmDriver


from ising.base import IsingModel
from ising.samplers import SwendsenWangSampler, SingleSpinFlipSampler
from ising.observables import Energy, Magnetization
import argparse

#for handler in logging.root.handlers[:]:
#    logging.root.removeHandler(handler)


import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def single_run(args):
    
    if args.sampler == "ClusterFlip":
        sampler = SwendsenWangSampler
    elif args.sampler == "SingleSpinFlip":
        sampler = SingleSpinFlipSampler
    else:
        raise ValueError("Sampler not implemented")

    observables = [Energy(Jx = args.Jx, Jy = args.Jy), Magnetization()]

    # Initialize simulation
    IM = IsingModel(sampler=sampler, size = args.size, warmup = args.warmup, temp = args.T, Jx = args.Jx, Jy = args.Jy) 
    IM.simulate(steps = args.steps, observables = observables, sampling_frequency = args.sampling_frequency)
    trajectory_path = os.path.join(args.snapshot_dir, f"ising-trajectory_size={args.size}_T={round(args.T, 2)}_sampler={args.sampler}.h5")
    IM.save(trajectory_path, overwrite=True)

def ensemble_run(args):
    if args.slurm_config:
        slurm_config = ConfigParser(args.slurm_config)
        slurm_args = slurm_config.config_data

    driver = SlurmDriver()
    for T in np.arange(args.T_min, args.T_max + args.dT, args.dT):
        args.T = T
        excluded_args = ['T_min', 'T_max', 'dT', 'config', 'mode', 'slurm_config' ,'logging_dir']

        cmd_args = " ".join([f"--{k} {v}" for k, v in vars(args).items() if v is not None and k not in excluded_args])
        cmd = f"ising_model single_run {cmd_args}"
        logger.info(f"Submitting job {cmd}")
        slurm_args["job_name"] = f"ising_{args.size}_{args.T:.2f}_{args.sampler}"
        slurm_args["output_dir"] = args.logging_dir
        driver.submit_job(cmd, env="thermomaps-ising", slurm_args=slurm_args, track=True)

    #driver.wait()

def parse_args():
    parser = argparse.ArgumentParser(description='Run Ising model simulation.')
    
    # Create subparsers for the two modes
    subparsers = parser.add_subparsers(dest='mode')

    # Single run parser
    single_parser = subparsers.add_parser('single_run', help='Run a single simulation')
    single_parser.add_argument('--sampler', type=str, default="ClusterFlip", help='Specifies the sampling method to be used in the simulation. Default is "ClusterFlip".')
    single_parser.add_argument('--size', type=int, default=8, help='Defines the size of the lattice for the simulation.')
    single_parser.add_argument('--T', type=float, default=1.0, help='Sets the temperature of the system for the simulation.')
    single_parser.add_argument('--Jx', type=int, default=1, help='Specifies the strength of the spin-spin interaction along the x-axis.')
    single_parser.add_argument('--Jy', type=int, default=1, help='Specifies the strength of the spin-spin interaction along the y-axis.')
    single_parser.add_argument('--warmup', type=int, default=1000, help='Number of warmup steps before the actual simulation begins.')
    single_parser.add_argument('--steps', type=int, default=10000, help='Total number of steps to run in the simulation.')
    single_parser.add_argument('--sampling_frequency', type=int, default=1, help='Frequency at which the system state is sampled and recorded.')
    single_parser.add_argument('--snapshot_dir', type=str, default="", help='Directory path where snapshots of the simulation will be stored.')
    single_parser.add_argument('--config', type=str, required=False, help='Optionally read arguments from config file.')

    # Ensemble run parser
    ensemble_parser = subparsers.add_parser('ensemble_run', help='Run an ensemble of simulations with varying parameters.')
    ensemble_parser.add_argument('--sampler', type=str, default="ClusterFlip", help='Specifies the sampling method to be used in the ensemble simulations. Default is "ClusterFlip".')
    ensemble_parser.add_argument('--size', type=int, default=8, help='Defines the size of the lattice for the simulations in the ensemble.')
    ensemble_parser.add_argument('--T_min', type=float, default=1.0, help='Sets the minimum temperature for the range of temperatures in the ensemble.')
    ensemble_parser.add_argument('--T_max', type=float, default=2.0, help='Sets the maximum temperature for the range of temperatures in the ensemble.')
    ensemble_parser.add_argument('--dT', type=float, default=0.5, help='Temperature step size between simulations in the ensemble.')
    ensemble_parser.add_argument('--Jx', type=int, default=1, help='Specifies the strength of the spin-spin interaction along the x-axis for the ensemble simulations.')
    ensemble_parser.add_argument('--Jy', type=int, default=1, help='Specifies the strength of the spin-spin interaction along the y-axis for the ensemble simulations.')
    ensemble_parser.add_argument('--warmup', type=int, default=1000, help='Number of warmup steps before each simulation in the ensemble begins.')
    ensemble_parser.add_argument('--steps', type=int, default=10000, help='Total number of steps to run in each simulation of the ensemble.')
    ensemble_parser.add_argument('--sampling_frequency', type=int, default=1, help='Frequency at which the system state is sampled and recorded in each simulation of the ensemble.')
    ensemble_parser.add_argument('--snapshot_dir', type=str, default="", help='Directory path where snapshots of the ensemble simulations will be stored.')
    ensemble_parser.add_argument('--logging_dir', type=str, default="", help='Directory path where snapshots of the ensemble simulations will be stored.')
    ensemble_parser.add_argument('--config', type=str, required=False, help='Optionally read arguments from a config file.')
    ensemble_parser.add_argument('--slurm_config', type=str, required=False, help='Optionally read slurm arguments from a config file.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.config:
        config = ConfigParser(args.config)
        args = config.override_args(args)
    

    # Determine which function to call based on the subcommand used
    if args.mode == 'single_run':
        single_run(args)
    elif args.mode == 'ensemble_run':
        ensemble_run(args)
    else:
        raise ValueError("Invalid mode selected. Choose 'single_run' or 'ensemble_run'.")

if __name__ == "__main__":
    main()
