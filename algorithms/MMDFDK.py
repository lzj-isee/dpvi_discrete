import torch
from ._funcs import duplicate_kill_particles
from algorithms.MMDF import MMDF


class MMDFDK(MMDF):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)
    
    @torch.no_grad()
    def one_step_update(self, step_size = None, alpha = None, tgt_support = None, tgt_mass = None, **kw):
        _beta = kw['beta']
        temp_support = self.particles + torch.randn_like(self.particles) * _beta
        first_var, vector_field = MMDF.first_variation_vector_field(temp_support, self.mass, tgt_support, tgt_mass, self.knType, self.bwType, self.bwVal)
        self.particles = self.particles - step_size * vector_field
        avg_first_var = first_var - torch.sum(first_var * self.mass)
        prob_list = 1 - torch.exp( - avg_first_var.abs() * alpha * step_size)
        self.particles = duplicate_kill_particles(prob_list, avg_first_var > 0, self.particles, noise_amp = self.opts.noise_amp, mode = 'parallel')

    def get_state(self):
        return self.particles, self.mass