import argparse
import itertools
import os
import shutil

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optimizer
import torchvision
from torch.multiprocessing import Process

from models.networks import MsImageDis, CRAVIM
from core.base_dataset import UnalignedDataset
from core.util import Logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


def recon_criterion(input, target):
    return torch.mean(torch.abs(input - target))


def gen_img_by_denoise(args, network, x, y, coeff):
    with torch.no_grad():
        t = torch.randint(0, args.num_timesteps, (args.batch_size,), device=x.device)
        y_t, y_tp = q_sample_pairs(coeff, y, t)
        predict = network(torch.cat([x, y_tp], dim=1))
    return predict[:, [1], :]


# train
def train(rank, gpu, args):
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    dataset = UnalignedDataset(opt=args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              sampler=train_sampler,
                                              drop_last=True)

    netG_A2B = CRAVIM(input_nc=args.input_nc * 2,
                      output_nc=args.input_nc * 2,
                      ).to(device)
    broadcast_params(netG_A2B.parameters())

    netG_B2A = CRAVIM(input_nc=args.input_nc * 2,
                      output_nc=args.input_nc * 2,
                      ).to(device)
    broadcast_params(netG_B2A.parameters())

    optimizerG = optimizer.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=args.lr_g, betas=(args.beta1_g, args.beta2_g))
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)

    netD_A = MsImageDis(args).to(device)  # discriminator for domain a
    broadcast_params(netD_A.parameters())

    netD_B = MsImageDis(args).to(device)  # discriminator for domain b
    broadcast_params(netD_B.parameters())

    optimizerD = optimizer.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()),
                                lr=args.lr_d, betas=(args.beta1_d, args.beta2_d))
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # ----------------------------------------------------------------------------------------------------------------

    exp_path = "./results/{}/train_info".format(args.exp)

    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)

    if args.resume:
        if os.path.exists('./results/{}/train_info/netG_A2B_{}.pth'.format(args.exp, args.resume_epoch)):
            ckpt = torch.load('./results/{}/train_info/netG_A2B_{}.pth'.format(args.exp, args.resume_epoch),
                              map_location=device)
            netG_A2B.load_state_dict(ckpt)

        if os.path.exists('./results/{}/train_info/netG_B2A_{}.pth'.format(args.exp, args.resume_epoch)):
            ckpt = torch.load('./results/{}/train_info/netG_B2A_{}.pth'.format(args.exp, args.resume_epoch),
                              map_location=device)
            netG_B2A.load_state_dict(ckpt)

        if os.path.exists('./results/{}/train_info/netD_A_{}.pth'.format(args.exp, args.resume_epoch)):
            ckpt = torch.load('./results/{}/train_info/netD_A_{}.pth'.format(args.exp, args.resume_epoch),
                              map_location=device)
            netD_A.load_state_dict(ckpt)

        if os.path.exists('./results/{}/train_info/netD_B_{}.pth'.format(args.exp, args.resume_epoch)):
            ckpt = torch.load('./results/{}/train_info/netD_B_{}.pth'.format(args.exp, args.resume_epoch),
                              map_location=device)
            netD_B.load_state_dict(ckpt)

        init_epoch = args.resume_epoch + 1
        global_step, epoch = 0, 0
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    logger = Logger(args.exp, '6006', args.num_epoch, len(data_loader))

    coeff = Diffusion_Coefficients(args, device)

    for epoch in range(init_epoch + 1, args.num_epoch + 1):
        train_sampler.set_epoch(epoch)

        for iteration, data in enumerate(data_loader):
            x_a = data["A"].to(device, non_blocking=True)
            x_b = data["B"].to(device, non_blocking=True)
            # -----------------------------------训练MUNIT生成器---------------------------------
            optimizerG.zero_grad()
            noise_t = torch.ones(size=(args.batch_size,), device=device, dtype=torch.int64) * (args.num_timesteps - 1)
            _, noise1 = q_sample_pairs(coeff, torch.randn_like(x_a), noise_t)
            _, noise2 = q_sample_pairs(coeff, torch.randn_like(x_b), noise_t)

            x_ab = netG_A2B(torch.cat([x_a, noise1], dim=1))  # G_A(A)
            x_ab_a, x_ab = x_ab[:, [0], :], x_ab[:, [1], :]

            x_ba = netG_B2A(torch.cat([x_b, noise2], dim=1))  # G_B(B)
            x_ba_b, x_ba = x_ba[:, [0], :], x_ba[:, [1], :]

            t1 = torch.randint(0, args.num_timesteps, (args.batch_size,), device=device)
            t2 = torch.randint(0, args.num_timesteps, (args.batch_size,), device=device)

            a_t, a_tp = q_sample_pairs(coeff, x_a, t1)
            b_t, b_tp = q_sample_pairs(coeff, x_b, t2)

            x_a_diff = netG_B2A(torch.cat([x_ab.detach(), a_tp.detach()], dim=1))
            x_a_diff_ab, x_a_diff = x_a_diff[:, [0], :], x_a_diff[:, [1], :]

            x_b_diff = netG_A2B(torch.cat([x_ba.detach(), b_tp.detach()], dim=1))
            x_b_diff_ba, x_b_diff = x_b_diff[:, [0], :], x_b_diff[:, [1], :]

            loss_gen_rec_a = recon_criterion(x_a_diff, x_a)
            loss_gen_rec_b = recon_criterion(x_b_diff, x_b)

            loss_trans1_a = recon_criterion(x_ab_a, x_a)
            loss_trans1_b = recon_criterion(x_ba_b, x_b)

            loss_trans2_ab = recon_criterion(x_a_diff_ab, x_ab)
            loss_trans2_ba = recon_criterion(x_b_diff_ba, x_ba)

            x_ab_dn = gen_img_by_denoise(args, netG_A2B, x_a, x_ab, coeff)
            x_ba_dn = gen_img_by_denoise(args, netG_B2A, x_b, x_ba, coeff)

            loss_rdn_ab = recon_criterion(x_ab, x_ab_dn)
            loss_rdn_ba = recon_criterion(x_ba, x_ba_dn)

            loss_gen_adv_a = netD_A.calc_gen_loss(x_ba)
            loss_gen_adv_b = netD_B.calc_gen_loss(x_ab)

            errG = ((loss_gen_adv_a + loss_gen_adv_b)
                    + 5 * (loss_rdn_ab + loss_rdn_ba)
                    + (loss_trans1_a + loss_trans1_b)
                    + (loss_trans2_ab + loss_trans2_ba)
                    + 15 * (loss_gen_rec_a + loss_gen_rec_b)
                    )
            errG.backward()
            optimizerG.step()
            # -----------------------------------训练MUNIT鉴别器---------------------------------
            optimizerD.zero_grad()
            errD_A = netD_A.calc_dis_loss(x_ba.detach(), x_a)
            errD_B = netD_B.calc_dis_loss(x_ab.detach(), x_b)

            errD = errD_A + errD_B
            errD.backward()
            optimizerD.step()
            # ----------------------------------------------------------------------------------
            global_step += 1
            logger.log(
                {
                    "adv": loss_gen_adv_a + loss_gen_adv_b,
                    "cycle": loss_gen_rec_a + loss_gen_rec_b,
                    "trans1": loss_trans1_a + loss_trans1_b,
                    "trans2": loss_trans2_ab + loss_trans2_ba,
                    "rdn": loss_rdn_ab + loss_rdn_ba,
                },
                images={
                    "real_A": x_a,
                    "real_B": x_b,
                },
            )

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if rank == 0:
            torchvision.utils.save_image(torch.cat
                                         ([x_a, x_a_diff, x_ab, x_b, x_b_diff, x_ba],
                                          dim=-1),
                                         os.path.join(exp_path, 'predict_epoch_{}.png'.format(epoch)),
                                         normalize=True)

            if epoch % args.save_ckpt_every == 0:
                torch.save(netG_A2B.state_dict(), os.path.join(exp_path, 'netG_A2B_{}.pth'.format(epoch)))
                torch.save(netG_B2A.state_dict(), os.path.join(exp_path, 'netG_B2A_{}.pth'.format(epoch)))
                # torch.save(netD_A.state_dict(), os.path.join(exp_path, 'netD_A_{}.pth'.format(epoch)))
                # torch.save(netD_B.state_dict(), os.path.join(exp_path, 'netD_B_{}.pth'.format(epoch)))


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '8097'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='gloo', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train parameters')
    parser.add_argument('--exp', default='IXI_CRAVIM', help='name of experiment')
    parser.add_argument('--dataroot', default='../Datasets/IXI', help='name of dataset')
    parser.add_argument('--phase', default='train', help='train or val')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--max_dataset_size', default=float("inf"))
    parser.add_argument('--serial_batches', default=True, help='if true, takes images in order to make batches')
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')

    parser.add_argument('--resume', default=False)
    parser.add_argument('--resume_epoch', type=int, default=50)

    parser.add_argument('--image_size', type=int, default=256, help='size of image')
    parser.add_argument('--num_channels', type=int, default=1, help='channel of image for model')
    parser.add_argument('--input_nc', type=int, default=1, help='ichannel of image for dataloader')
    parser.add_argument('--num_workers', type=int, default=8, help='num of dataloader')
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--NFE', type=int, default=4)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')

    # setting of MUNIT Dis
    parser.add_argument('--dim', type=int, default=64, help='number of filters in the bottommost layer')
    parser.add_argument('--pad_type', type=str, default="reflect")
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--gan_type', type=str, default="lsgan")
    parser.add_argument('--norm', type=str, default="none")
    parser.add_argument('--dis_activ', type=str, default="lrelu")
    parser.add_argument('--num_scales', type=int, default=3)

    # setting of noise
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)

    # generator and training
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=2e-4, help='learning rate g')
    parser.add_argument('--beta1_g', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2_g', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1_d', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2_d', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay', default=True)
    parser.add_argument('--r1_gamma', type=float, default=1, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=10, help='lazy regulariation.')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')

    # ddp
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        init_processes(0, size, train, args)
