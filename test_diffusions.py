import sys
sys.path.append("/root/dev/jjuke_diffusion")
import argparse
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from jjuke_diffusion.diffusion.common import get_betas
from jjuke_diffusion.diffusion.ddim import DDIMSampler
from jjuke_diffusion.diffusion.ddpm import DDPMSampler, DDPMTrainer
from jjuke_diffusion.diffusion.karras import KarrasSampler
from jjuke_diffusion.unet.unet_base import UnetBase

from jjuke.models.ema_trainer import ema
from jjuke.models.scheduler import LinearWarmup, WarmupScheduler
from jjuke.metrics.score import get_inception_and_fid_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="/root/hdd1/diffusion_test_mine")
    parser.add_argument("--eval", action="store_true") # for shell script mode
    parser.add_argument("--ema", action="store_true") # for shell script mode
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--n_samples_eval", type=int, default=1000)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--samples_per_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--n_sampler_steps", type=int, default=50)
    parser.add_argument("--model_mean_type", default="eps") # ["eps", "x_start", "x_prev"]
    parser.add_argument("--loss_type", default="l2") # ["l2", "rescaled_l2", "l1", "rescaled_l1", "kl", "rescaled_kl"]
    parser.add_argument("--model_var_type", default="fixed_small") # ["fixed_small", "fixed_large", "learned", "learned_range"]
    parser.add_argument("--dataset_dir", default="/root/hdd1/CIFAR10")
    parser.add_argument("--dataset_stats_path", default="/root/hdd1/CIFAR10/stats", help="Path to CIFAR10 dataset stats")
    args = parser.parse_args()

    # for debugging mode
    # args.result_dir = "/root/hdd1/unet_test"
    # args.eval = False
    # args.ema = False
    
    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def model_params(model):
    model_size = 0
    for param in model.parameters():
        if param.requires_grad:
            model_size += param.data.nelement()
    return model_size


def infinite_dataloader(dl: DataLoader, n_steps: int = np.inf):
    i = 0
    alive = True
    while alive:
        for batch in dl:
            yield batch
            i += 1
            if i >= n_steps:
                alive = False
                break


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt

    def get(self):
        return self.avg

    def __call__(self):
        return self.avg


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def calc_score(ims: Tensor, args):
    """
    ### input:
    - ims: b 3 h w, cpu
    """
    (IS, IS_std), FID = get_inception_and_fid_score(
        ims,
        fid_cache=os.path.join(args.dataset_stats_path, "cifar10.train.npz"),
        use_torch=False,
        verbose=True,
    )
    return IS, IS_std, FID


def train(args, model: nn.Module, model_ema: nn.Module):
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    # sched = WarmupScheduler(optim, args.warmup)
    sched = LinearWarmup(optim, args.warmup, args.n_steps, 0.05)

    betas = get_betas("linear", 1000)
    trainer = DDPMTrainer(
        betas,
        model_mean_type=args.model_mean_type,
        model_var_type=args.model_var_type,
        loss_type=args.loss_type,
    ).cuda()
    sampler = KarrasSampler(
        betas,
        karras_num_timesteps=args.n_sampler_steps,
        karras_sampler="heun",
        model_var_type=args.model_var_type,
        clip_denoised=True
    ).cuda()
    # sampler = DDIMSampler(
    #     betas,
    #     ddim_num_timesteps=args.n_sampler_steps,
    #     eta=0.,
    #     model_mean_type="eps",
    #     model_var_type=args.model_var_type,
    #     clip_denoised=True,
    # ).cuda()
    # sampler = DDPMSampler(betas, model_mean_type="eps", model_var_type=args.model_var_type, clip_denoised=True).cuda()
    # trainer = GaussianDiffusionTrainer(model, 1e-4, 2e-2, 1000).cuda()
    # sampler = GaussianDiffusionSampler(model, 1e-4, 2e-2, 1000).cuda()

    if args.rankzero:
        ds_train = CIFAR10(args.dataset_dir, train=True, download=False)
    ds_train = CIFAR10(args.dataset_dir, train=True, transform=Compose([RandomHorizontalFlip(), ToTensor()]))
    dl_kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    dl_train = infinite_dataloader(DataLoader(ds_train, shuffle=True, **dl_kwargs), n_steps=args.n_steps)

    o = AverageMeter()
    with tqdm(total=args.n_steps, ncols=100, disable=not args.rankzero, desc="Train") as pbar:
        for step, (im, label) in enumerate(dl_train, 1):
            im: Tensor = im.cuda(non_blocking=True) * 2 - 1  # [0, 1] -> [-1, 1]
            # label: Tensor = label.cuda(non_blocking=True)

            optim.zero_grad()
            losses = trainer(model, im)
            loss = losses["loss"].mean()
            # loss = trainer(im).mean()
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            if args.ddp:
                ema(model.module, model_ema, 0.9999)
            else:
                ema(model, model_ema, 0.9999)

            o.update(loss.item(), n=im.size(0))
            pbar.set_postfix_str(f"loss: {o():.4f}", refresh=False)
            pbar.update()

            if step % args.samples_per_steps == 0:
                model.eval()
                with torch.no_grad():
                    # save sample
                    if args.ddp:
                        n = args.n_samples
                        m = math.ceil(n / dist.get_world_size())
                        samples = sampler(model.module, (m, 3, 32, 32)) / 2 + 0.5  # [-1, 1] -> [0, 1]
                        # samples = sampler((m, 3, 32, 32)) / 2 + 0.5
                        samples_lst = [torch.empty_like(samples) for _ in range(args.world_size)]
                        dist.all_gather(samples_lst, samples)
                        samples = torch.cat(samples_lst)[:n]
                    else:
                        samples = sampler(model, (args.n_samples, 3, 32, 32)) / 2 + 0.5  # [-1, 1] -> [0, 1]

                    if args.rankzero:
                        save_image(samples, os.path.join(args.sample_dir, f"{step:06d}.png"), nrow=int(math.sqrt(args.n_samples)))

                        # save checkpoint
                        state_dict = {
                            "model": model.module.state_dict() if args.ddp else model.state_dict(),
                            "model_ema": model_ema.state_dict(),
                        }
                        torch.save(state_dict, os.path.join(args.result_dir, "best.pth"))

                # pbar.clear()
                print(flush=True)

                args.ema = False
                eval(args, model, model_ema)
                args.ema = True
                eval(args, model, model_ema)

                model.train()

    if args.ddp:
        dist.barrier()


@torch.no_grad()
def eval(args, model: nn.Module, model_ema: nn.Module):
    model.eval()
    model_ema.eval()

    if args.ema:
        model = model_ema
    else:
        if args.ddp:
            model = model.module
        else:
            model = model

    betas = get_betas("linear", 1000)

    # Karras Sampler -> best score!
    sampler = KarrasSampler(
        betas,
        karras_num_timesteps=args.n_sampler_steps,
        karras_sampler="heun", # ["heun", "dpm", "ancestral"]
        model_var_type=args.model_var_type,
        clip_denoised=True
    ).cuda()
    # DDIM Sampler
    # sampler = DDIMSampler(
    #     betas,
    #     ddim_num_timesteps=args.n_sampler_steps,
    #     eta=0.,
    #     model_mean_type="eps",
    #     model_var_type=args.model_var_type,
    #     clip_denoised=True,
    # ).cuda()
    # DDPM Sampler
    # sampler = DDPMSampler(
    #     betas,
    #     model_mean_type="eps",
    #     model_var_type=args.model_var_type,
    #     clip_denoised=True
    # ).cuda()

    # generate images
    n = args.n_samples_eval
    m = math.ceil(n / args.world_size)
    batch_size = args.batch_size

    ims = []
    with tqdm(total=n, ncols=100, disable=not args.rankzero, desc="Eval") as pbar:
        for i in range(0, m, batch_size):
            b = min(m - i, batch_size)
            x: Tensor = sampler(model, (b, 3, 32, 32))
            # x: Tensor = sampler((b, 3, 32, 32))
            x = x.div_(2).add_(0.5).clamp_(0, 1)  # [-1, 1] -> [0, 1]

            if args.ddp:
                xs = [torch.empty_like(x) for _ in range(args.world_size)]
                dist.all_gather(xs, x)
                x = torch.cat(xs)
            if args.rankzero:
                ims.append(x.cpu())

            pbar.update(min(pbar.total - pbar.n, b * args.world_size))

    # calculate FID
    if args.rankzero:
        ims = torch.cat(ims)
        IS, IS_std, FID = calc_score(ims, args)
        print(f"IS: {IS:.4f}, IS_std: {IS_std:.4f}, FID: {FID:.4f}")

    if args.ddp:
        dist.barrier()


def main_worker(rank: int, args: argparse.Namespace):
    if args.ddp:
        dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=rank)

    args.rank = rank
    args.rankzero = rank == 0
    args.gpu = args.gpus[rank]
    torch.cuda.set_device(args.gpu)
    seed_everything(args.rank)

    if args.ddp:
        print(f"main_worker with rank:{rank} (gpu:{args.gpu}) is loaded", torch.__version__)
    else:
        print(f"main_worker with gpu:{args.gpu} in main thread is loaded", torch.__version__)

    args.result_dir = os.path.join(args.result_dir, "UnetBase_DDPMTrainer_KarrasSampler_fixed_small".format(
        "UnetBase", "DDPMTrainer", "KarrasSampler",
        args.loss_type, args.model_var_type
    ))
    args.sample_dir = os.path.join(args.result_dir, "samples")
    args.output_dir = os.path.join(args.result_dir, "outputs")
    if args.rankzero:
        os.makedirs(args.sample_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

    out_channels = 6 if args.model_var_type.startswith("learned") else 3

    model = UnetBase(
        unet_dim=2,
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4, 8) # 64 -> 128 -> 256 -> 512
    ).cuda()
    # model = UNetModel(
    #     in_channels=3,
    #     out_channels=3,
    #     channels=64,
    #     n_res_blocks=2,
    #     attention_levels=[0, 1, 2, 3],
    #     channel_multipliers=[1, 2, 4, 8], # 64 -> 128 -> 256 -> 512
    #     n_heads=4, # attn_heads in UnetBase
    #     tf_layers=2, # Num of "Blocks" in UnetBase
    #     d_cond=0 # 0 means no conditioning
    # ).cuda()

    n_model_params = model_params(model)
    print("Model Params: %.2fM" % (n_model_params / 1e6))

    model_ema: nn.Module = deepcopy(model)
    if args.ddp:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_ema.load_state_dict(model.module.state_dict())
    else:
        model_ema.load_state_dict(model.state_dict())
    model_ema.eval().requires_grad_(False)

    if not args.eval:
        print("Training...")
        train(args, model, model_ema)
    else:
        print("Evaluation...")
        ckpt = torch.load(os.path.join(args.result_dir, "best.pth"), map_location="cpu")
        if args.ddp:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["model_ema"])

        eval(args, model, model_ema)


def main():
    args = get_args()

    args.gpus = list(map(int, args.gpus.split(",")))
    args.world_size = len(args.gpus)
    args.ddp = args.world_size > 1

    if args.ddp:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        pc = mp.spawn(main_worker, nprocs=args.world_size, args=(args,), join=False)
        pids = " ".join(map(str, pc.pids()))
        print("\33[101mProcess Ids:", pids, "\33[0m")
        try:
            pc.join()
        except KeyboardInterrupt:
            print("\33[101mkill %s\33[0m" % pids)
            os.system("kill %s" % pids)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    main()