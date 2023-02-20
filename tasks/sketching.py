import torch, os, numpy as np, matplotlib.pyplot as plt, ot
from geomloss import SamplesLoss


class sketching(object):
    def __init__(self, opts, **kw) -> None:
        self.opts = opts
        self.device = opts.device
        self.obj_dataloader = kw['obj_dataloader']
        self.aspect_hw = self.obj_dataloader.aspect_hw
        self.tgt_support = self.obj_dataloader.tgt_support
        self.tgt_mass = self.obj_dataloader.tgt_mass
        self.original_pix = self.obj_dataloader.original
        self.record_support = []
        self.record_mass = []
        self.record_sinkdiv = []
        self.loss_op = SamplesLoss(
            loss = 'sinkhorn', p = 2, blur = 0.01, backend = 'online', scaling = 0.95
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
        self.record_support.append(support.cpu().numpy())
        self.record_mass.append(mass.cpu().numpy())
        self.record_sinkdiv.append(sinkdiv)
        writer.add_scalar('sinkhorn_div', self.record_sinkdiv[-1], global_step = count)
        logger.info('count: {}, sinkhorn_div: {:.2e}'.format(count, self.record_sinkdiv[-1]))
        fig = self.plot_result(support, mass, (self.opts.img_size, int(self.opts.img_size * self.aspect_hw)), self.device)
        plt.savefig(os.path.join(save_folder, '%s_%d.png'%(self.opts.dataset, count)), bbox_inches = 'tight')
        plt.close()
        

    @torch.no_grad()
    def final_process(self, support, mass, writer, logger, save_folder, is_save):
        # save the target image
        fig = self.plot_result(self.tgt_support, self.tgt_mass, (len(torch.unique(self.tgt_support[:, 0])), len(torch.unique(self.tgt_support[:, 1]))), device = self.device)
        plt.savefig(os.path.join(save_folder, 'target.png'), bbox_inches = 'tight')
        plt.close()
        # save the final result to tensorboard
        fig = self.plot_result(support, mass, (self.opts.img_size, int(self.opts.img_size * self.aspect_hw)), device = self.device)
        writer.add_figure(tag = 'hist', figure = fig)
        plt.close()
        # save the loss
        np.save(os.path.join(save_folder, 'sinkhorn_div.npy'), np.array(self.record_sinkdiv))
        if is_save:
            np.save(os.path.join(save_folder, 'support.npy'), np.array(self.record_support))
            np.save(os.path.join(save_folder, 'mass.npy'), np.array(self.record_mass))

    def plot_result(self, support, _mass, bins: tuple, device):
        fig = plt.figure()
        ax = fig.gca()
        plt.clf()
        plt.axes().set_aspect(self.aspect_hw)
        support_x = support[:, 0].cpu().numpy()
        support_y = support[:, 1].cpu().numpy()
        mass = _mass.clone()
        mass[mass < 0] = 0
        mass = mass.cpu().numpy()
        plt.hist2d(support_x, support_y, bins = bins, weights = mass)
        plt.axis('off')
        return fig