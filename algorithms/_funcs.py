import torch
import numpy as np

def gaussian_stein_kernel(x: torch.Tensor, y, score_x, score_y, bw): 
    # x: required_grad: False, y: required_grad: True
    _, p = x.shape
    sq_dist = x.pow(2).sum(1).view(-1, 1) + y.pow(2).sum(1).view(1, -1) - 2 * torch.matmul(x, y.t())
    kernel = ( - sq_dist / bw).exp()
    part1 = torch.matmul(score_x, score_y.t()) * kernel
    part2 = 2 * ((score_x * x).sum(1, keepdim = True) - torch.matmul(score_x, y.t())) * kernel / bw
    part3 = -2 * (torch.matmul(x, score_y.t()) - (y * score_y).sum(1)) * kernel / bw
    part4 = (2 * p / bw - 4 * sq_dist / bw**2) * kernel
    ksd_matrix = part1 + part2 + part3 + part4
    return ksd_matrix

def duplicate_kill_particles(prob_list, kill_list: torch.Tensor, particles: torch.Tensor, noise_amp, mode = 'parallel'):
    # will modify the input particles
    rand_number = torch.rand(particles.shape[0], device = particles.device)
    index_list = torch.linspace(0, particles.shape[0] - 1, particles.shape[0], dtype = torch.int, device = particles.device)
    if mode == 'sequential':
        rand_index = torch.randint(0, particles.shape[0] - 1, (particles.shape[0],), device = particles.device)
        for k in range(particles.shape[0]):
            if kill_list[k]: # kill particle k, duplicate with random noise 
                if rand_number[k] < prob_list[k]:
                    particles[k] = particles[index_list != k][rand_index[k]].clone() + torch.randn(particles.shape[1], device = particles.device) * noise_amp
            else: # duplicate particle k, duplicate with random noise
                if rand_number[k] < prob_list[k]:
                    particles[index_list != k][rand_index[k]] = particles[k].clone() + torch.randn(particles.shape[1], device = particles.device) * noise_amp
        return particles
    elif mode == 'parallel':
        unchange_particles = particles[(rand_number >= prob_list)]
        duplicate_particles = particles[torch.bitwise_and(rand_number < prob_list, torch.logical_not(kill_list))]
        new_particles = torch.cat([unchange_particles, duplicate_particles, duplicate_particles + torch.randn_like(duplicate_particles) * noise_amp], dim = 0)
        if new_particles.shape[0] == particles.shape[0]:
            pass
        elif new_particles.shape[0] < particles.shape[0]: # duplicate randomly
            rand_index = torch.randint(0, new_particles.shape[0], (particles.shape[0] - new_particles.shape[0], ), device = new_particles.device)
            new_particles = torch.cat([new_particles, new_particles[rand_index] + torch.randn_like(new_particles[rand_index]) * noise_amp], dim = 0)
        else: # kill randomly
            rand_index = torch.randperm(new_particles.shape[0], device = new_particles.device)
            new_particles = new_particles[rand_index][:particles.shape[0]].clone()
        assert new_particles.shape[0] == particles.shape[0], 'change the particle number!'
        return new_particles
    else:
        raise NotImplementedError

def safe_reciprocal(x):
    return 1. / torch.maximum(x, torch.ones(1, dtype = x.dtype, device = x.device) * 1e-16)

def safe_log(x):
    # return torch.log(torch.maximum(torch.ones_like(x) * 1e-32, x))
    return torch.log(1e-16 + x)

def kernel_func(particles, knType = 'rbf', bwType = 'med', bwVal = 1, bw_only = False):
    particle_num = particles.shape[0]
    cross_diff = particles[:, None, :] - particles[None, :, :]
    sq_distance = torch.sum(cross_diff.pow(2), dim = 2)
    if bwType == 'med': # SVGD
        bw_h = torch.median(sq_distance + 1e-5) / np.log(particle_num)
    elif bwType == 'nei': # GFSD, Blob
        bw_h = sq_distance + torch.diag(torch.diag(sq_distance) + sq_distance.max())
        bw_h = bw_h.min(dim = 1)[0].mean() if particle_num > 1 else 1
    elif bwType == 'fix': # fixed bandwidth
        bw_h = bwVal
    elif bwType == 'heu': # this bandwith is from Mirrored SVGD
        n_elems = sq_distance.shape[0] * sq_distance.shape[1]
        topk_values = torch.topk(sq_distance.view(-1), k = n_elems // 2, sorted = False).values
        bw_h = torch.min(topk_values)
        bw_h = torch.where(bw_h == 0, torch.ones_like(bw_h), bw_h)
    else: 
        raise NotImplementedError
    if bw_only: return None, None, bw_h
    if knType == 'imq': 
        kernel = (1 + sq_distance / bw_h).pow(-0.5)
        nabla_kernel = -kernel.pow(3)[:, :, None] * cross_diff / bw_h
    elif knType == 'rbf':
        kernel = (-sq_distance / bw_h).exp()
        nabla_kernel = -2 * cross_diff * kernel[:, :, None] / bw_h
    else:
        raise NotImplementedError
    return kernel, nabla_kernel, bw_h

def kernel_func_v2(par1, par2, knType = 'rbf', bwType = 'med', bwVal = 1, bw_only = False):
    cross_diff = par1[:, None, :] - par2[None, :, :]
    sq_dist = torch.sum(cross_diff.pow(2), dim = 2)
    if bwType == 'fix':
        bw_h = bwVal
    elif bwType == 'med':
        bw_h = torch.median(sq_dist + 1e-5) / np.log(sq_dist.shape[1])
    elif bwType == 'nei':
        _temp = sq_dist.clone()
        _temp[_temp < 1e-6] = sq_dist.max()
        bw_h = _temp.min(dim = 1)[0].mean() 
    else:
        raise NotImplementedError
    if bw_only: return None, None, bw_h
    if knType == 'imq':
        kernel = (1 + sq_dist / bw_h).pow(-0.5)
        nabla_kernel = -kernel.pow(3)[:, :, None] * cross_diff / bw_h
    elif knType == 'rbf':
        kernel = (-sq_dist / bw_h).exp()
        nabla_kernel = -2 * cross_diff * kernel[:, :, None] / bw_h
    else:
        raise NotImplementedError
    return kernel, nabla_kernel, bw_h 