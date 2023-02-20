import torch, os, numpy as np, matplotlib.pyplot as plt, ot, torchvision.transforms as transforms, torchvision
from geomloss import SamplesLoss


class color_transfer(object):
    def __init__(self, opts, **kw) -> None:
        self.opts = opts
        self.device = opts.device
        self.obj_dataloader = kw['obj_dataloader']
        self.tgt_support = self.obj_dataloader.tgt_support
        self.tgt_mass = self.obj_dataloader.tgt_mass
        self.record_support = []
        self.record_mass = []
        self.record_sinkdiv = []
        self.record_w2 = []
        self.loss_op = SamplesLoss(
            loss = 'sinkhorn', p = 2, blur = 0.01, backend = 'online', scaling = 0.95
        )

    @torch.no_grad()
    def init_support(self):
        return self.obj_dataloader.src_support

    @torch.no_grad()
    def init_mass(self):
        return self.obj_dataloader.src_mass

    @torch.no_grad()
    def evaluation(self, support, mass, writer, logger, count: int, save_folder):
        # evaluate the wasserstein-2 distance
        '''
        cost_matrix = torch.cdist(support, self.tgt_support).pow(2)
        transport = ot.emd(mass, self.tgt_mass, cost_matrix)
        w2_value = (cost_matrix * transport).sum().sqrt().item()
        '''
        w2_value = 0
        # evaluate sinkhorn divergence
        sinkdiv = self.loss_op(
            mass, support, self.tgt_mass, self.tgt_support
        ).item()
        self.record_support.append(support.cpu().numpy())
        self.record_mass.append(mass.cpu().numpy())
        self.record_sinkdiv.append(sinkdiv)
        self.record_w2.append(w2_value)
        writer.add_scalar('sinkhorn_div', self.record_sinkdiv[-1], global_step = count)
        writer.add_scalar('w2', self.record_w2[-1], global_step = count)
        logger.info('count: {}, sd: {:.2e}, w2: {:.2e}'.format(count, self.record_sinkdiv[-1], self.record_w2[-1]))
        image = self.plot_result(support, mass)
        torchvision.utils.save_image(
            image, os.path.join(save_folder, '%s_%d.png'%(self.opts.dataset, count)), normalize = True
        )

    @torch.no_grad()
    def final_process(self, support, mass, writer, logger, save_folder, is_save):
        # save the target image
        image = self.plot_result(self.tgt_support, self.tgt_mass)
        torchvision.utils.save_image(
            image, os.path.join(save_folder, 'target.png'), normalize = True
        )
        # save the final result to tensorboard
        image = self.plot_result(support, mass)
        writer.add_image(tag = 'image', img_tensor = image)
        # save the loss
        np.save(os.path.join(save_folder, 'sinkhorn_div.npy'), np.array(self.record_sinkdiv))
        np.save(os.path.join(save_folder, 'w2.npy'), np.array(self.record_w2))
        if is_save:
            np.save(os.path.join(save_folder, 'support.npy'), np.array(self.record_support))
            np.save(os.path.join(save_folder, 'mass.npy'), np.array(self.record_mass))

    def plot_result(self, support: torch.Tensor, _mass: torch.Tensor):
        image = support.transpose(0, 1).view(3, int(np.sqrt(support.shape[0])), int(np.sqrt(support.shape[0]))).cpu()
        return transforms.functional.resize(image, (self.obj_dataloader.image_size, self.obj_dataloader.image_size), transforms.InterpolationMode.BICUBIC)