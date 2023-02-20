import pretty_errors
from functools import partial
import argparse
import importlib
import torch
import os
import numpy as np
import utils
from dataloader import myDataLoader

def main(opts):
    utils.set_random_seed(opts.seed)
    writer, logger, save_folder = utils.get_logger(opts, name = opts.algorithm)
    data_and_loader = myDataLoader(opts)
    task = importlib.import_module('tasks.%s'%opts.task).__getattribute__(opts.task)(opts, obj_dataloader = data_and_loader)
    algorithm = importlib.import_module('algorithms.%s'%opts.algorithm).__getattribute__(opts.algorithm)(
        opts, task.init_support(), 
        torch.ones(opts.particle_num, device = opts.device) / opts.particle_num if opts.task not in ['color_transfer', 'morphing'] else task.init_mass() 
    )
    for step in range(opts.max_iter):
        if hasattr(opts, 'alpha'):
            alpha = np.tanh((2.0 * step / opts.max_iter)**5) * opts.alpha if opts.al_warmup else opts.alpha
        else:
            alpha = 0.0
        if hasattr(opts, 'beta'):
            cur_beta = (1 - step / opts.max_iter) * opts.beta
            cur_beta = max(cur_beta, opts.beta_min)
        else:
            cur_beta = None
        # one step update
        algorithm.one_step_update(
            step_size = opts.lr, # common param
            alpha = alpha, # valid in CA/DK type methods
            tgt_support = task.tgt_support, # common param
            tgt_mass = task.tgt_mass, # common param
            beta = cur_beta # valid in MMD flow based method
        )
        support, mass = algorithm.get_state()
        utils.check(support, mass, step, logger)
        # evaluation
        if step % opts.eval_interval == 0 or (step + 1) == opts.max_iter:
            task.evaluation(support, mass, writer = writer, logger = logger, count = step, save_folder = save_folder)
    task.final_process(
        support, mass, writer = writer, logger = logger, save_folder = save_folder, 
        is_save = opts.save_particles if hasattr(opts, 'save_particles') else None)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--algorithm', type = str, default = 'SD') 
    parser.add_argument('--task', type = str, default = 'flow')
    parser.add_argument('--dataset', type = str, default='cifar10')
    parser.add_argument('--max_iter', type = int, default = 200)
    parser.add_argument('--eval_interval', type = int, default = 20)
    parser.add_argument('--save_folder', type = str, default='results')
    parser.add_argument('--device', type = str, default = 'cuda:3')
    parser.add_argument('--seed', type = int, default = 9, help = 'random seed for algorithm')
    parser.add_argument('--split_size', type = float, default = 0.1, help = 'split ratio for dataset')
    parser.add_argument('--split_seed', type = int, default = 19, help = 'random seed to split dataset')
    parser.add_argument('--save_particles', action = 'store_true')
    opts,_ = parser.parse_known_args()
    # algorithm setting
    if opts.algorithm in ['SD', 'MMDF', 'SDCA', 'SDDK', 'MMDFCA', 'MMDFDK']:
        parser.add_argument('--particle_num', type = int, default = 2000)
        parser.add_argument('--lr', type = float, default = 1.0)
        if opts.algorithm in ['MMDF', 'MMDFCA', 'MMDFDK']:
            parser.add_argument('--beta', type = float, default = 0, help = 'noise level in MMD flow')
            parser.add_argument('--beta_min', type = float, default = 0.003)
            parser.add_argument('--knType', type = str, default = 'rbf', choices = ['rbf', 'imq'], help = 'type of kernel function')
            parser.add_argument('--bwType', type = str, default = 'fix', choices = ['med', 'heu', 'nei', 'fix'])
            parser.add_argument('--bwVal', type = float, default = 0.1)
        if opts.algorithm in ['SD', 'SDCA', 'SDDK']:
            parser.add_argument('--power', type = int, default = 2)
            parser.add_argument('--blur', type = float, default = 0.01, help = 'https://www.kernel-operations.io/geomloss/api/pytorch-api.html')
            parser.add_argument('--scaling', type = float, default = 0.95, help = 'refer to the above link')
            parser.add_argument('--backend', type = str, default = 'online', help = 'refer to the above link')
        if opts.algorithm in ['SDCA', 'SDDK', 'MMDFCA', 'MMDFDK']:
            parser.add_argument('--alpha', type = float, default = 1.0)
            parser.add_argument('--al_warmup', action = 'store_true')
        if opts.algorithm in ['SDDK', 'MMDFDK']:
            parser.add_argument('--noise_amp', type = float, default = 0.01)
    opts,_ = parser.parse_known_args()
    # task setting
    if opts.task in ['sketching']:
        parser.add_argument('--img_size', type = int, default = 100, help = 'ouput: len of width')
    if opts.task in ['morphing']:
        parser.add_argument('--resize', type = int, default = 200, help = 'the size of image before discretized to particles')
        parser.add_argument('--max_src_num', type = int, default = 1000, help = 'maximum number of particles of source distribution')
        parser.add_argument('--max_tgt_num', type = int, default = 2000, help = 'maximum number of particles of target distribution')
        parser.add_argument('--hgradient', action = 'store_true', help = 'whether reweight the mass of target distribution')
        parser.add_argument('--plot_size', type = int, default = 10, help = 'the size of particles in scatter')
    if opts.task in ['color_transfer']:
        parser.add_argument('--square_size', type = int, default = 0)
        parser.add_argument('--src_downscale', type = int, default = 1, choices = [1, 2, 4, 8, 16], help = 'downscale for source image')
        parser.add_argument('--tgt_downscale', type = int, default = 8, choices = [1, 2, 4, 8, 16])
    if opts.task in ['flow']:
        parser.add_argument('--tgt_num', type = int, default = 2000, help = 'the number of particles of target distribution')
    opts = parser.parse_args()
    main(opts)