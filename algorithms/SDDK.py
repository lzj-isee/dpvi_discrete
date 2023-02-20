import torch, numpy as np
from geomloss import SamplesLoss
from ._funcs import kernel_func, safe_log, duplicate_kill_particles


class SDDK(object):
    def __init__(self, opts, init_particles, init_mass) -> None:
        self.opts = opts
        self.particles = init_particles
        self.mass = init_mass
        self.potential_op = SamplesLoss(
            loss = 'sinkhorn', p = opts.power, blur = opts.blur, potentials = True, 
            debias = False, backend = opts.backend, scaling = opts.scaling
        )
    
    def one_step_update(self, step_size = None, alpha = None, tgt_support = None, tgt_mass = None, **kw):
        self.particles.requires_grad = True
        first_var_ab, _ = self.potential_op(
            self.mass, self.particles, tgt_mass, tgt_support
        )
        first_var_aa, _ = self.potential_op(
            self.mass, self.particles, self.mass, self.particles 
        )
        first_var_ab = first_var_ab.view(-1)
        first_var_aa = first_var_aa.view(-1)
        first_var_ab_grad = torch.autograd.grad(
            torch.sum(first_var_ab), self.particles
        )[0]
        first_var_aa_grad = torch.autograd.grad(
            torch.sum(first_var_aa), self.particles
        )[0]
        with torch.no_grad():
            vector_field = first_var_ab_grad - first_var_aa_grad
            first_var = first_var_ab - first_var_aa
            # update the positions of particles
            self.particles = self.particles - step_size * vector_field
            # duplicate and kill particles
            avg_first_var = first_var - torch.sum(first_var * self.mass)
            prob_list = 1 - torch.exp( - avg_first_var.abs() * alpha * step_size)
            self.particles = duplicate_kill_particles(prob_list, avg_first_var > 0, self.particles, noise_amp = self.opts.noise_amp, mode = 'parallel')

        self.particles.requires_grad = False

    def get_state(self):
        return self.particles, self.mass