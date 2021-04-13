# -*- coding: utf-8 -*-
# @Date    : 2/16/21
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0
import torch
import torch.nn as nn
from torchvision.utils import make_grid


def adjust_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_func(
    args,
    network: nn.Module,
    optimizer,
    style_iter,
    content_iter,
    device,
    num_iter,
    save_dir,
    writer=None,
):
    network.train()
    adjust_learning_rate(args, optimizer, iteration_count=num_iter)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if writer:
        writer.add_scalar("loss_content", loss_c.item(), num_iter + 1)
        writer.add_scalar("loss_style", loss_s.item(), num_iter + 1)

        if (
            (num_iter + 1) % args.save_model_interval == 0
            or (num_iter + 1) == args.max_iter
            or num_iter == 0
        ):
            if args.saadain:
                states = {
                    "decoder_state_dict": network.decoder.state_dict(),
                    "saadain_state_dict": network.style_norm.state_dict(),
                }
            else:
                states = {
                    "decoder_state_dict": network.decoder.state_dict(),
                }
            torch.save(states, save_dir / "ckpt_iter_{:d}.pth.tar".format(num_iter + 1))
            with torch.no_grad():
                if writer:
                    output = network.style_transfer(content_images, style_images, 1.0)
                    styled_img_grid = make_grid(
                        output, nrow=4, normalize=True, scale_each=True
                    )
                    reference_img_grid = make_grid(
                        style_images, nrow=4, normalize=True, scale_each=True
                    )
                    content_img_grid = make_grid(
                        content_images, nrow=4, normalize=True, scale_each=True
                    )

                    writer.add_image("styled_images", styled_img_grid, num_iter)
                    writer.add_image("reference_images", reference_img_grid, num_iter)
                    writer.add_image("content_images", content_img_grid, num_iter)


@torch.no_grad()
def val_func(
    args, network: nn.Module, style_iter, content_iter, device, num_iter, writer=None
):
    print("=> validating...")
    network.eval()
    content_losses = []
    style_losses = []

    while True:
        try:
            content_images = next(content_iter).to(device)
            style_images = next(style_iter).to(device)
            loss_c, loss_s = network(content_images, style_images)
            loss_c = args.content_weight * loss_c
            loss_s = args.style_weight * loss_s
            content_losses.append(loss_c)
            style_losses.append(loss_s)
        except StopIteration:
            break

    avg_loss_c = sum(content_losses) / len(content_losses)
    avg_loss_s = sum(style_losses) / len(style_losses)
    print(f"validate a total of {len(content_losses)} iterations")
    if writer:
        writer.add_scalar("val/avg_loss_content", avg_loss_c.item(), num_iter + 1)
        writer.add_scalar("val/avg_loss_style", avg_loss_s.item(), num_iter + 1)
