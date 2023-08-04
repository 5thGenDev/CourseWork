# Copyright (c) EEEM071, University of Surrey

import torch


def init_lr_scheduler(
    optimizer,
    lr_scheduler="multi_step",  # learning rate scheduler
    stepsize=[20, 40],  # step size to decay learning rate
    gamma=0.1,  # learning rate decay
    epochs,
):
    warmup_lr_scheduler = = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=5
    )
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - 5, eta_min=args.lr_min
    )
    # what is lr_min????
    
    if lr_scheduler == "single_step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize[0], gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == "sequential":
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[5]
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
