from __future__ import annotations

import glob
import math
import os
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from detection import setup_paths
from detection.test import test
from detection.yolov3.models import Darknet, YOLOLayer, init_seeds, load_darknet_weights
from detection.yolov3.utils.datasets import Dataset
from detection.yolov3.utils.torch_utils import ModelEMA, select_device
from detection.yolov3.utils.utils import (
    compute_loss,
    fitness,
    labels_to_class_weights,
    labels_to_image_weights,
    load_classes,
    plot_images,
    plot_results,
    strip_optimizer,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Dict, Optional, Tuple


mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print("Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex")
    mixed_precision = False  # not installed


def train(
    train_dataset: Dataset,
    test_dataset: Dataset,
    path_dict: Dict[str, Path],
    hyp: Dict[str, Any],
    opt: Dict[str, Any],
    tb_writer: Optional[SummaryWriter] = None,
) -> Tuple[float]:
    global mixed_precision

    # Set device
    device = select_device(opt["device"], apex=mixed_precision, batch_size=opt["batch_size"])
    if device.type == "cpu":
        mixed_precision = False

    cfg = path_dict["cfg_file"]
    epochs = opt["epochs"]  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt["batch_size"]
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = path_dict["start_weights_file"]  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt["img_size"]  # img sizes (min, max, test)
    classes = load_classes(path_dict["names_file"])
    results_file = path_dict["results_file"]

    checkpoint_folder = path_dict["weights_folder"] / (opt["name"] + "_ckpt")
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    # Image Sizes
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_min, gs)
    opt["multi_scale"] |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt["multi_scale"]:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    img_size = imgsz_max  # initialize with max size

    # Configure run
    init_seeds()

    nc = 1 if opt["single_cls"] else int(len(classes))  # number of classes
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

    # Remove previous results
    for f in glob.glob("*_batch*.jpg") + glob.glob(str(results_file)):
        os.remove(f)

    # Initialize model
    model = Darknet(str(cfg)).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if ".bias" in k:
            pg2 += [v]  # biases
        elif "Conv2d.weight" in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt["adam"]:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = torch.optim.Adam(pg0, lr=hyp["lr0"])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = torch.optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    print("Optimizer groups: %g .bias, %g Conv2d.weight, %g other" % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0

    if weights.suffix == ".pt":  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = (
                "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. "
                "See https://github.com/ultralytics/yolov3/issues/657" % (weights, cfg, weights)
            )
            raise KeyError(s) from e

        # load optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # load results
        if ckpt.get("training_results") is not None:
            results_file.write_text(ckpt["training_results"])

        # epochs
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print(
                "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                % (weights, ckpt["epoch"], epochs)
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    if opt["freeze_layers"]:
        output_layer_indices = [
            idx - 1 for idx, module in enumerate(model.module_list) if isinstance(module, YOLOLayer)
        ]
        freeze_layer_indices = [
            x
            for x in range(len(model.module_list))
            if (x not in output_layer_indices) and (x - 1 not in output_layer_indices)
        ]
        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training
    if device.type != "cpu" and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        torch.distributed.init_process_group(
            backend="nccl",  # 'distributed backend'
            init_method="tcp://127.0.0.1:9999",  # distributed training init method
            world_size=1,  # number of nodes for distributed training
            rank=0,
        )  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataloader
    batch_size = min(batch_size, len(train_dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nw,
        shuffle=not opt["rect"],  # Shuffle=True unless rectangular training is used
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    # Testloader
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # Model EMA
    ema = ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)
    maps = np.zeros(nc)  # mAP per class
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print("Image sizes %g - %g train, %g test" % (imgsz_min, imgsz_max, imgsz_test))
    print("Using %g dataloader workers" % nw)
    print("Starting training for %g epochs..." % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if train_dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(train_dataset.labels, nc=nc, class_weights=w)
            train_dataset.indices = random.choices(
                range(train_dataset.n), weights=image_weights, k=train_dataset.n
            )  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        print(("\n" + "%10s" * 8) % ("Epoch", "gpu_mem", "GIoU", "obj", "cls", "total", "targets", "img_size"))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Burn-in
            if ni <= n_burn:
                xi = [0, n_burn]  # x interp
                model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, 64 / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    x["weight_decay"] = np.interp(ni, xi, [0.0, hyp["weight_decay"] if j == 1 else 0.0])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, hyp["momentum"]])

            # Multi-Scale
            if opt["multi_scale"]:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = torch.nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
            s = ("%10s" * 2 + "%10.3g" * 6) % ("%g/%g" % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # Plot
            if ni < 1:
                f = "train_batch%g.jpg" % i  # filename
                res = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, res, dataformats="HWC", global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        if not opt["notest"] or final_epoch:  # Calculate mAP
            results, maps = test(
                testloader,
                ema.ema,
                path_dict,
                batch_size=batch_size,
                imgsz=imgsz_test,
                save_json=final_epoch,
                single_cls=opt["single_cls"],
                multi_label=ni > n_burn,
            )

        # Write
        with results_file.open("a") as f:
            f.write(s + "%10.3g" * 7 % results + "\n")  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt["name"]) and opt["bucket"]:
            os.system("gsutil cp results.txt gs://%s/results/results%s.txt" % (opt["bucket"], opt["name"]))

        # Tensorboard
        if tb_writer:
            tags = [
                "train/giou_loss",
                "train/obj_loss",
                "train/cls_loss",
                "metrics/precision",
                "metrics/recall",
                "metrics/mAP_0.5",
                "metrics/F1",
                "val/giou_loss",
                "val/obj_loss",
                "val/cls_loss",
            ]
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt["nosave"]) or (final_epoch and not opt["evolve"])
        if save:
            with results_file.open("r") as f:  # create checkpoint
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "training_results": f.read(),
                    "model": ema.ema.module.state_dict() if hasattr(model, "module") else ema.ema.state_dict(),
                    "optimizer": None if final_epoch else optimizer.state_dict(),
                }

            # Save every 10 epochs (starting from epoch 9)
            if epoch % 10 == 9:
                torch.save(ckpt, checkpoint_folder / "ckpt_{}.pt".format(epoch))

            # Save last, best and delete
            torch.save(ckpt, path_dict["last_weights_file"])
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, path_dict["best_weights_file"])
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Rename results file
    n = opt["name"]
    n = "_" + n if not n.isnumeric() else n
    results_file.rename("data/training_results%s.txt" % n)

    # Strip optimizer
    for pt_file in path_dict["weights_folder"].glob("*.pt"):
        strip_optimizer(str(pt_file))
        if opt["bucket"]:
            os.system("gsutil cp %s gs://%s/weights" % (str(pt_file), opt["bucket"]))
                
    if not opt["evolve"]:
        plot_results()  # save as results.png
    print("%g epochs completed in %.3f hours.\n" % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    torch.distributed.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results
