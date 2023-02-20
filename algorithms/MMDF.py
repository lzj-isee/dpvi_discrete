import torch
from ._funcs import kernel_func, kernel_func_v2


class MMDF(object):
    def __init__(self, opts, init_particles, init_mass) -> None:
        self.opts = opts
        self.knType = opts.knType
        self.bwType = opts.bwType
        self.bwVal = opts.bwVal
        self.particles = init_particles
        self.mass = init_mass

    @classmethod
    def first_variation_vector_field(cls, src_support, src_mass, tgt_support, tgt_mass, knType, bwType, bwVal):
        kernel_11, nabla_kernel_11, _ = kernel_func_v2(src_support, src_support, knType, bwType, bwVal, bw_only = False)
        kernel_12, nabla_kernel_12, _ = kernel_func_v2(src_support, tgt_support, knType, bwType, bwVal, bw_only = False)
        first_variation = (kernel_11 * src_mass[None, :]).sum(dim = 1) - (kernel_12 * tgt_mass[None, :]).sum(dim = 1)
        vector_field = (nabla_kernel_11 * src_mass[None, :, None]).sum(dim = 1) - (nabla_kernel_12 * tgt_mass[None, :, None]).sum(dim = 1)
        return first_variation, vector_field
    
    @torch.no_grad()
    def one_step_update(self, step_size = None, tgt_support = None, tgt_mass = None, **kw):
        _beta = kw['beta']
        temp_support = self.particles + torch.randn_like(self.particles) * _beta
        _, vector_field = MMDF.first_variation_vector_field(temp_support, self.mass, tgt_support, tgt_mass, self.knType, self.bwType, self.bwVal)
        self.particles = self.particles - step_size * vector_field

    def get_state(self):
        return self.particles, self.mass