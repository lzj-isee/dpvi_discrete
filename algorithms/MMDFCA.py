import torch
from algorithms.MMDF import MMDF


class MMDFCA(MMDF):
    def __init__(self, opts, init_particles, init_mass) -> None:
        super().__init__(opts, init_particles, init_mass)
    
    @torch.no_grad()
    def one_step_update(self, step_size = None, alpha = None, tgt_support = None, tgt_mass = None, **kw):
        _beta = kw['beta']
        temp_support = self.particles + torch.randn_like(self.particles) * _beta
        first_var, vector_field = MMDF.first_variation_vector_field(temp_support, self.mass, tgt_support, tgt_mass, self.knType, self.bwType, self.bwVal)
        self.particles = self.particles - step_size * vector_field
        self.mass *= 1 - step_size * alpha * (first_var - torch.sum(first_var * self.mass))
        self.mass = self.mass / self.mass.sum() # eliminate numerical error

    def get_state(self):
        return self.particles, self.mass