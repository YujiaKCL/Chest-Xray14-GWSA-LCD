import os
import torch
import torch.distributed as dist
import random
import numpy as np
import warnings
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning

# warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning, module='sklearn')

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def setup_random_seed(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def calc_acc(logits, targets):
    preds = torch.sigmoid(logits).ge(0.5).float()
    acc = (targets == preds).sum() / len(targets)
    return acc

def calc_auc(logits, targets):
    if torch.isnan(logits).any():
        print('logits:', logits)
    if torch.isnan(targets).any():
        print('targets:', targets)

    targets = targets.cpu()
    preds = torch.sigmoid(logits).cpu().detach()
    fpr, tpr, thresholds = metrics.roc_curve(targets, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def load_trained_model(model, save_path):
    checkpoint = torch.load(save_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)

def load_checkpoint(args, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {args.resume} ....................")
    checkpoint = torch.load(args.resume, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'].module, strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler: lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    args.start_epoch = checkpoint['epoch'] + 1
    logger.info(f"=> loaded successfully resume from '{args.resume}' (epoch {checkpoint['epoch']})")

def save_checkpoint(args, epoch, model, optimizer, lr_scheduler, logger):
    dict_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                  'epoch': epoch,
                  'config': args}

    save_path = os.path.join(args.ckpt_path, args.tag, f'ckpt_{args.backbone}.pth')
    os.makedirs(os.path.join(args.ckpt_path, args.tag), exist_ok=True)
    torch.save(dict_state, save_path)
    logger.info(f"State dict saved to {save_path} !")
