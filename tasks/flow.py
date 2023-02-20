import torch, os, numpy as np, matplotlib.pyplot as plt, ot, torchvision
from geomloss import SamplesLoss


class flow(object):
    def __init__(self, opts, **kw) -> None:
        self.opts = opts
        self.device = opts.device
        self.obj_dataloader = kw['obj_dataloader']
        self.tgt_support = self.obj_dataloader.tgt_support
        self.tgt_mass = self.obj_dataloader.tgt_mass
        self.record_support = []
        self.record_mass = []
        self.record_sinkdiv = []
        self.loss_op = SamplesLoss(
            loss = 'sinkhorn', p = opts.power, blur = opts.blur, backend = 'online', scaling = opts.scaling 
        )

    @torch.no_grad()
    def init_support(self):
        result = torch.rand(self.opts.particle_num, self.tgt_support.shape[1], device = self.device)
        return result

    @torch.no_grad()
    def evaluation(self, support, mass, writer, logger, count: int, save_folder):
        # evaluate sinkhorn divergence
        sinkdiv = self.loss_op(
            mass, support, self.tgt_mass, self.tgt_support
        ).item()
        if self.opts.save_particles:
            self.record_support.append(support.cpu().numpy())
            self.record_mass.append(mass.cpu().numpy())
        self.record_sinkdiv.append(sinkdiv)
        writer.add_scalar('sinkhorn_div', self.record_sinkdiv[-1], global_step = count)
        logger.info('count: {}, sinkhorn_div: {:.2e}'.format(count, self.record_sinkdiv[-1]))
        images = self.convert2image(support)
        torchvision.utils.save_image(images, os.path.join(save_folder, '%s_%d.jpg'%(self.opts.dataset, count)))


    @torch.no_grad()
    def final_process(self, support, mass, writer, logger, save_folder, is_save):
        # save the image
        images = self.convert2image(support)
        torchvision.utils.save_image(images, os.path.join(save_folder, 'final.png'))
        # save the loss
        np.save(os.path.join(save_folder, 'sinkhorn_div.npy'), np.array(self.record_sinkdiv))
        if is_save:
            np.save(os.path.join(save_folder, 'support.npy'), np.array(self.record_support))
            np.save(os.path.join(save_folder, 'mass.npy'), np.array(self.record_mass))

    def convert2image(self, support, num = 8):
        image_size = self.obj_dataloader.image_size
        support = support[:num ** 2].view(num ** 2, 3, image_size[0], image_size[1])
        images = torchvision.utils.make_grid(support, nrow = num, value_range = (0, 1), normalize = True)
        return images