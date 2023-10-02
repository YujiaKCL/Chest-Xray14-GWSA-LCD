import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import AverageMeter, accuracy
from timm.models import model_parameters
from utils import calc_acc, calc_auc, reduce_tensor

np.set_printoptions(precision=4, linewidth=200)


def train_epoch(model, device, train_loader, criterion, optimizer, logger, epoch):
    model.train()
    num_classes = train_loader.dataset.num_classes
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    losses = np.zeros(num_classes)
    accs = np.zeros(num_classes)
    aucs = np.zeros(num_classes)
    count = 0
    for idx, (imgs, targets) in enumerate(train_loader):
        bz = targets.size(0)
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = 0.
        for i in range(num_classes):
            weight = None
            if targets[:, i].sum() > 0:
                weight = (bz / targets[:, i].sum()) * targets[:, i] + (bz / (bz - targets[:, i].sum())) * (1 - targets[:, i])
            loss_i = F.binary_cross_entropy_with_logits(outputs[:, i], targets[:, i], weight=weight)

            loss += loss_i
            losses[i] = (losses[i] * count + loss_i.item() * bz) / (count + bz)

            acc_i = calc_acc(outputs[:, i], targets[:, i])
            accs[i] = (accs[i] * count + acc_i.item() * bz) / (count + bz)

            if targets[:, i].sum() > 0:
                auc_i = calc_auc(outputs[:, i], targets[:, i])
                aucs[i] = (aucs[i] * count + auc_i.item() * bz) / (count + bz)

        count += bz

        if idx % 100 == 0 or idx == len(train_loader) - 1:
            logger.info(f'Epoch: {epoch} batch: {idx} \n Loss: {losses} \n Accuracy: {accs} Avg: {accs.mean():.4f} \n AUCs: {aucs} Avg: {aucs.mean():.4f}')
            
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        loss.backward(create_graph=second_order)
        optimizer.step()

        torch.cuda.synchronize()

    eval_metrics = {'Loss': losses, 'Accuracy': accs, 'AUC': aucs}
    return eval_metrics


@torch.no_grad()
def val_epoch(model, device, data_loader, criterion):
    model.eval()
    num_classes =  data_loader.dataset.num_classes
    y_true = []
    y_logits = []
    losses = torch.zeros(num_classes).to(device)
    count = 0
    for idx, (imgs, targets) in enumerate(data_loader):
        bz = targets.size(0)
        y_true.append(targets)
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(imgs)
        y_logits.append(outputs.cpu())
        for i in range(num_classes):
            loss_i = criterion(outputs[:, i], targets[:, i])
            losses[i] = (losses[i] * count + loss_i.item() * bz) / (count + bz)
        count += bz
    
    y_true = torch.cat(y_true, 0)
    y_logits = torch.cat(y_logits, 0)
    accs = torch.tensor([calc_acc(y_logits[:, i], y_true[:, i]).item() for i in range(num_classes)]).to(device)
    aucs = torch.tensor([calc_auc(y_logits[:, i], y_true[:, i]) for i in range(num_classes)]).to(device)

    torch.cuda.synchronize()
    losses = reduce_tensor(losses)
    accs = reduce_tensor(accs)
    aucs = reduce_tensor(aucs)

    eval_metrics = {'Loss': losses.cpu().numpy(), 'Accuracy': accs.cpu().numpy(), 'AUC': aucs.cpu().numpy()}
    return eval_metrics


@torch.no_grad()
def eval_matrics(model, device, data_loader, metrics):
    model.eval()
    y_true, y_logits = [], []
    for idx, (imgs, targets) in enumerate(data_loader):
        bz = targets.size(0)
        y_true.append(targets)
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(imgs)
        y_logits.append(outputs.cpu())
    
    y_true = torch.cat(y_true, 0).to(torch.long)
    y_logits = torch.cat(y_logits, 0)
    
    eval_result = {}
    for metric in metrics:
        eval_result[metric] = metrics[metric](torch.nn.functional.sigmoid(y_logits), y_true)
    
    return eval_result
