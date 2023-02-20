import torch
from geomloss import SamplesLoss


class SDCA(object):
    def __init__(self, opts, init_particles, init_mass) -> None:
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
            self.particles = self.particles - step_size * vector_field
            self.mass *= 1 - step_size * alpha * (first_var - torch.sum(first_var * self.mass))
            self.mass = self.mass / self.mass.sum() # eliminate numerical error

        self.particles.requires_grad = False

    def get_state(self):
        return self.particles, self.mass