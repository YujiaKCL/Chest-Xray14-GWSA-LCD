import os
import argparse
from contextlib import suppress
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from timm.optim import create_optimizer, create_optimizer_v2
from timm.scheduler import create_scheduler
from logger import create_logger
from timm.loss import BinaryCrossEntropy
from models import build_model
from utils import setup_random_seed, load_checkpoint, save_checkpoint
from data import ChestXray14Dataset, build_transform, calc_class_dist
from engine import train_epoch, val_epoch, eval_matrics
from torchmetrics.classification import MultilabelRecall, MultilabelSpecificity, MultilabelAccuracy, MultilabelAUROC


def parse_args():
    parser = argparse.ArgumentParser('Code for training classifiers for Chest-Xray14 Dataset')

    # Dataset parameters
    parser.add_argument('--data', type=str, 
                        default='~/datasets/chest-xray14', 
                        help='path to ChestXray14 dataset')
    parser.add_argument('--split', type=str, choices=['official', 'non-official'],
                        default='official',
                        help='Official: split the dataset according to given splits, \
                        Non-official: the given split is not reasonable in terms of data distribution, so split it by ourselves.')
    parser.add_argument('--has-val', action='store_true',
                        help='If has-val, 1/8 training set will be into validation set')
    parser.set_defaults(has_val=True)
    parser.add_argument('--img-size', default=512, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # Model parameters
    parser.add_argument('--backbone', default='densenet121',help='model backbone')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--pretrain', action='store_true',
                        help='If True, load the pretrained weights for backbone')

    # Model modification
    parser.add_argument('--attention', action='store_true',
                        help='If True, append GWSA module to the model')
    parser.add_argument('--correlation', action='store_true',
                        help='If True, append LCD module to the model')

    parser.add_argument('--seed', default=0, type=int, help='random seed for reproduction')

    # Training parameters
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='weight decay (default: 0)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='multistep', type=str, metavar='SCHEDULER',
                        choices=['cosine', None, 'plateau', 'multistep'],
                        help='LR scheduler (default: "cosine"')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-milestones', type=list, default=[5, 10, 15],
                        help='milestones for decay lr for MultiStepLRScheduler')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=2, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 5')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--step-on-epochs', action='store_true',
                        help='Timm scheduler, turn on to adjust lr every epoch')
    parser.set_defaults(step_on_epochs=True)

    # Log and Checkpoints parameters
    parser.add_argument('--tag', default='default', type=str,
                        help='root folder for experimental recording')
    parser.add_argument('--log-output', default='logs', type=str,
                        help='root of output folder for logs')
    parser.add_argument('--ckpt-path', default='checkpoints', type=str, 
                        help='Path to save model checkpoints')
    parser.add_argument('--save-freq', default=1, type=int, 
                        help='Frequency for saving state dict in epoch')
    parser.add_argument('--resume', type=str, 
                        help='Checkpoint path for recovering training')

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    
    args, unparsed = parser.parse_known_args()
    return args
    
def main(args):
    device = torch.device("cuda:{}".format(args.local_rank))

    logger.info('==> Building up datasets and loaders ...')
    train_transform = build_transform(train=True, args=args)
    test_transform = build_transform(train=False, args=args)
    
    train_set = ChestXray14Dataset(args.data, mode='train', split=args.split, has_val_set=args.has_val, transform=train_transform)
    val_set = ChestXray14Dataset(args.data, mode='valid', split=args.split, has_val_set=args.has_val, transform=test_transform) if args.has_val else None
    test_set = ChestXray14Dataset(args.data, mode='test', split=args.split, transform=test_transform)

    train_sampler = DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_mem, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_mem, drop_last=False, sampler=val_sampler) if val_set is not None else None
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_mem, drop_last=False, sampler=test_sampler)
    
    if args.has_val:
        logger.info(f'In total, {len(train_set)} training, {len(val_set)} validation, and {len(test_set)} testing samples \n \
                    From {train_set.num_patients}, {val_set.num_patients}, and {test_set.num_patients} patients. ')
    else:
        logger.info(f'In total, {len(train_set)} training and {len(test_set)} testing samples \n \
                    From {train_set.num_patients} and {test_set.num_patients} patients. ')

    logger.info(f'==> Creating model: {args.backbone} ...')
    num_classes = train_set.num_classes
    model = build_model(args.backbone, num_classes, args.pretrain, args.attention, args.correlation).cuda()
    optimizer = create_optimizer(args, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=False)
    model_without_ddp = model.module
    lr_scheduler, num_epochs = create_scheduler(args, optimizer) if hasattr(args, 'sched') else None
    criterion = BinaryCrossEntropy().cuda()

    if args.resume:
        load_checkpoint(args, model_without_ddp, optimizer, lr_scheduler, logger)

    if hasattr(model.module.fc, 'gamma'):
        logger.info(model.module.fc.gamma)

    logger.info(f'==> Starting training for {num_epochs} epochs from epoch {args.start_epoch} ...')
    for epoch in range(args.start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_metrics = train_epoch(model, device, train_loader, criterion, optimizer, logger, epoch)
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Training Epoch {epoch} lr: {lr} \n \
                    Loss: {train_metrics['Loss']} \n \
                    Accuracy: {train_metrics['Accuracy']} Avg: {train_metrics['Accuracy'].mean():.4f} \n \
                    AUCs: {train_metrics['AUC']} Avg: {train_metrics['AUC'].mean():.4f} \n")

        if val_loader is not None:
            val_metrics = val_epoch(model, device, val_loader, criterion)
            logger.info(f"Validation Epoch {epoch} \n \
                    Loss: {val_metrics['Loss']} \n \
                    Accuracy: {val_metrics['Accuracy']} Avg: {val_metrics['Accuracy'].mean():.4f} \n \
                    AUCs: {val_metrics['AUC']} Avg: {val_metrics['AUC'].mean():.4f} \n")

        if test_loader is not None:
            test_metrics = val_epoch(model, device, test_loader, criterion)
            logger.info(f"Test Epoch {epoch} \n \
                    Loss: {test_metrics['Loss']} \n \
                    Accuracy: {test_metrics['Accuracy']} Avg: {test_metrics['Accuracy'].mean():.4f} \n \
                    AUCs: {test_metrics['AUC']} Avg: {test_metrics['AUC'].mean():.4f} \n")
        
        if hasattr(model.module.fc, 'gamma'):
            logger.info(model.module.fc.gamma)
        
        if lr_scheduler:
            if args.sched == 'plateau':
                lr_scheduler.step(epoch, metric = val_metrics['AUC'].mean())
            else:
                lr_scheduler.step(epoch)

        if args.ckpt_path and dist.get_rank() == 0 and ((epoch + 1) % args.save_freq == 0 or epoch == num_epochs - 1):
            save_checkpoint(args, epoch, model_without_ddp, optimizer, lr_scheduler, logger)
            
    logger.info('Training ends ...')

    logger.info('==> Start testing on test set ...')
    test_metrics = val_epoch(model, device, test_loader, criterion)
    logger.info(f"Test \n \
                    Loss: {test_metrics['Loss']} \n \
                    Accuracy: {test_metrics['Accuracy']} Avg: {test_metrics['Accuracy'].mean():.4f} \n \
                    AUCs: {test_metrics['AUC']} Avg: {test_metrics['AUC'].mean():.4f} \n")
    
    metrics = {'Sensitivity': MultilabelRecall(test_set.num_classes, average=None, sync_on_compute=True), 
           'Specificity': MultilabelSpecificity(test_set.num_classes, average=None, sync_on_compute=True),
           'Accuracy': MultilabelAccuracy(test_set.num_classes, average=None, sync_on_compute=True),
           'AUROC': MultilabelAUROC(test_set.num_classes, average=None, sync_on_compute=True)}
    eval_result = eval_matrics(model, device, test_loader, metrics)
    for m in eval_result:
        logger.info(f"{m}: {eval_result[m]} Avg: {eval_result[m].mean()}")
        
    logger.info('Done.')

if __name__ == '__main__':
    args = parse_args()
    args.log_output = os.path.join(args.log_output, args.tag)
    os.makedirs(args.log_output, exist_ok=True)
    logger = create_logger(args.log_output, args.local_rank)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
    
    logger.info(f'==> Setting up random seed: {args.seed}')
    seed = args.seed + dist.get_rank()
    setup_random_seed(seed)
    logger.info('Enable Cudnn benchmark')
    torch.backends.cudnn.benchmark = True

    main(args)
    
    