import argparse
import os
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity
from core import html
from core.base_dataset import UnalignedDataset
from core.visualizer import save_images
from models.networks import CRAVIM


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


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_and_test(args):
    torch.manual_seed(args.seed)
    device = 'cuda:0'

    dataset = UnalignedDataset(opt=args)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.num_workers)

    task = "A2B"
    netG_A2B = CRAVIM(input_nc=args.input_nc * 2,
                      output_nc=args.input_nc * 2,
                      ).to(device)
    ckpt = torch.load('./results/{}/train_info/netG_{}_{}.pth'.format(args.exp, task, args.epoch_id),
                      map_location=device)

    netG_A2B.load_state_dict(ckpt)
    netG_A2B.eval()

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    web_dir = "./results/{}/{}_info/{}_{}_{}".format(args.exp, args.phase, args.phase, args.epoch_id, task)
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (args.exp, args.phase, args.epoch_id))

    with torch.no_grad():
        SSIM_score, PSNR_score, MAE_score, LPIPS_score = 0, 0, 0, 0
        for iteration, data in enumerate(tqdm(data_loader)):
            x = data["A"].to(device, non_blocking=True)
            t = torch.ones(size=(args.batch_size,), device=device, dtype=torch.int64) * (args.num_timesteps - 1)
            _, noise = q_sample_pairs(coeff, torch.randn_like(x), t)
            predict = noise
            for i in reversed(range(args.num_timesteps)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                predict_0 = netG_A2B(torch.cat([x, predict], dim=1))
                predict_0 = predict_0[:, [1], :]
                predict_new = sample_posterior(pos_coeff, predict_0, predict, t)
                predict = predict_new.detach()

            visuals = OrderedDict()
            visuals["real_A"] = data["A"]
            visuals["fake_B"] = predict
            visuals["real_B"] = data["B"]
            save_images(webpage, visuals, [str(iteration + 1).zfill(3)], width=256)

            real_B = Variable(data["B"]).detach().cpu()
            fake_B = predict.detach().cpu()

            real_B = real_B.numpy().squeeze()
            fake_B = fake_B.numpy().squeeze()

            ssim = structural_similarity(fake_B.squeeze(), real_B.squeeze(), data_range=(real_B.max() - real_B.min()))

            SSIM_score += ssim

        webpage.save()  # save the HTML
        length = iteration + 1
        print("对于A2B 总图片数:%d [平均SSIM分数:%f]" % (length, (SSIM_score / length)))

    task = "B2A"
    netG_B2A = CRAVIM(input_nc=args.input_nc * 2,
                      output_nc=args.input_nc * 2,
                      ).to(device)
    ckpt = torch.load('./results/{}/train_info/netG_{}_{}.pth'.format(args.exp, task, args.epoch_id),
                      map_location=device)

    netG_B2A.load_state_dict(ckpt)
    netG_B2A.eval()

    web_dir = "./results/{}/{}_info/{}_{}_{}".format(args.exp, args.phase, args.phase, args.epoch_id, task)
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (args.exp, args.phase, args.epoch_id))

    with torch.no_grad():
        SSIM_score, PSNR_score, MAE_score, LPIPS_score = 0, 0, 0, 0
        for iteration, data in enumerate(tqdm(data_loader)):
            x = data["B"].to(device, non_blocking=True)
            t = torch.ones(size=(args.batch_size,), device=device, dtype=torch.int64) * (args.num_timesteps - 1)
            _, noise = q_sample_pairs(coeff, torch.randn_like(x), t)
            predict = noise
            for i in reversed(range(args.num_timesteps)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                predict_0 = netG_B2A(torch.cat([x, predict], dim=1))
                predict_0 = predict_0[:, [1], :]
                predict_new = sample_posterior(pos_coeff, predict_0, predict, t)
                predict = predict_new.detach()

            visuals = OrderedDict()
            visuals["real_A"] = data["B"]
            visuals["fake_B"] = predict
            visuals["real_B"] = data["A"]
            save_images(webpage, visuals, [str(iteration + 1).zfill(3)], width=256)

            real_B = Variable(data["A"]).detach().cpu()
            fake_B = predict.detach().cpu()

            real_B = real_B.numpy().squeeze()
            fake_B = fake_B.numpy().squeeze()

            ssim = structural_similarity(fake_B.squeeze(), real_B.squeeze(), data_range=(real_B.max() - real_B.min()))

            SSIM_score += ssim
        webpage.save()  # save the HTML
        length = iteration + 1
        print("对于B2A 总图片数:%d [平均SSIM分数:%f]" % (length, (SSIM_score / length)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('test parameters')
    parser.add_argument('--exp', default='IXI_CRAVIM', help='name of experiment')
    parser.add_argument('--dataroot', default='../Datasets/IXI', help='name of dataset')
    parser.add_argument('--phase', default='test', help='train or val')
    parser.add_argument('--epoch_id', type=int, default=100)
    parser.add_argument('--max_dataset_size', default=float("inf"))
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--serial_batches', default=True,
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--image_size', type=int, default=256, help='size of image')
    parser.add_argument('--num_channels', type=int, default=1, help='channel of image for model')
    parser.add_argument('--input_nc', type=int, default=1, help='ichannel of image for dataloader')
    parser.add_argument('--num_workers', type=int, default=0, help='num of dataloader')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--z_emb_dim', type=int, default=64)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--num_channels_dae', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    args = parser.parse_args()
    sample_and_test(args)
