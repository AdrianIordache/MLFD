from common      import *
from config_file import CFG

def train(model, loader, optimizer, scheduler, criterion, epoch):
    model.train()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

        end = time.time()
        if CFG["use_wandb"]: 
                wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                            "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % CFG['print_freq'] == 0 or (batch + 1) == len(loader):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
        
    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    if CFG["use_wandb"]: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc

def validate(model, loader, criterion):
    model.eval()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    labels_, outputs_ = [], []
    start  = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)

        top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))
        loss = criterion(outputs, labels)

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

        end = time.time()
        if CFG["use_wandb"]: wandb.log({"valid_avg_acc@1": top1_accs.average, "valid_avg_acc@5" : top5_accs.average, "valid_avg_loss": losses.average})
        if (batch + 1) % CFG['print_freq'] == 0 or (batch + 1) == len(loader):
            message = f"[V] B: [{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f})"

            print(message)

    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    if CFG["use_wandb"]: wandb.log({"valid_epoch_acc@1": epoch_top1_acc, "valid_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc

def evaluate(model, loader, criterion):
    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    labels_, outputs_ = [], []
    start  = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)

        top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))
        loss = criterion(outputs, labels)

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    free_gpu_memory(DEVICE)
    return epoch_top1_acc, epoch_top5_acc