from common      import *
from config_file import *

def train(model, loader, optimizer, scheduler, criterion, epoch, mixup_fn, accumulation_steps, scaler):
    model.train()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if mixup_fn is not None:
            top1_acc, top5_acc = torch.tensor(-1), torch.tensor(-1)
        else:
            top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))

        if ((batch + 1) % accumulation_steps == 0) or (batch + 1 == len(loader)):
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

            if scheduler is not None: scheduler.step()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
    
    if mixup_fn is not None:
        epoch_top1_acc, epoch_top5_acc = torch.tensor(-1), torch.tensor(-1)
    else:
        epoch_outputs = torch.as_tensor(np.array(outputs_), dtype = torch.float32)
        epoch_labels  = torch.as_tensor(np.array(labels_))

        epoch_top1_acc, epoch_top5_acc = accuracy(epoch_outputs, epoch_labels, topk = (1, 5))
    
    if USE_WANDB: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()

def validate(model, loader, criterion, config):
    model.eval()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    labels_, outputs_ = [], []
    start  = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        # labels = labels + 300
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
        if USE_WANDB: wandb.log({"valid_avg_acc@1": top1_accs.average, "valid_avg_acc@5" : top5_accs.average, "valid_avg_loss": losses.average})
        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[V] B: [{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f})"

            print(message)

    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    if USE_WANDB: wandb.log({"valid_epoch_acc@1": epoch_top1_acc, "valid_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()

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

        end = time.time()
        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[E] B: [{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f})"

            print(message)
            
    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item(), epoch_top5_acc.item()


def train_kd(student, teacher, loader, optimizer, scheduler, criterion, epoch):
    teacher.train()
    student.train()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (images, embeddings, labels) in enumerate(loader):
        images     = images.to(DEVICE)
        labels     = labels.to(DEVICE)
        embeddings = embeddings.to(DEVICE)

        with torch.no_grad():
            teacher_logits = teacher(embeddings.float())

        student_logits     = student(images)
        top1_acc, top5_acc = accuracy(student_logits, labels, topk = (1, 5))

        loss = criterion(student_logits, teacher_logits, labels)        
        loss.backward()

        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(student_logits.detach().cpu().numpy())

        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
        
    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    if USE_WANDB: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()


def train_kd_conv(student, teacher, loader, optimizer, scheduler, criterion, epoch):
    def get_activation(name, out_dict):
        def hook(model, input, output):
            out_dict[name] = output
        return hook

    student_maps = {}
    student.model.layer4.register_forward_hook(get_activation('size_7', student_maps))

    student.train()
    teacher.eval()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()


    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (images, maps, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        maps   = maps.to(DEVICE)

        with torch.no_grad():
            teacher_logits, teacher_maps = teacher(maps, out_maps = True)

        student_logits     = student(images)
        top1_acc, top5_acc = accuracy(student_logits, labels, topk = (1, 5))

        loss = criterion(student_logits, teacher_logits, student_maps['size_7'], teacher_maps, labels)        
        loss.backward()

        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(student_logits.detach().cpu().numpy())

        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
        
    epoch_top1_acc, epoch_top5_acc = accuracy(torch.as_tensor(np.array(outputs_)), torch.as_tensor(np.array(labels_)), topk = (1, 5))
    if USE_WANDB: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()


def train_kd(student, teacher, loader, optimizer, scheduler, criterion, epoch, scaler, config):
    student_maps = {}
    
    if config["e_size"] == 7:
        student.model.layer4.register_forward_hook(get_activation('size_7',  student_maps))
        student.latent_proj.register_forward_hook(get_activation('size_1',  student_maps))

    student.train()
    teacher.eval()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    loss_maps = {
        "student_maps": None,
        "teacher_maps": None
    }

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (images, maps, labels) in enumerate(loader):
        # print(maps.shape)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        maps   = maps.to(DEVICE)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                student_logits = student(images)
        else:
            student_logits = student(images)

        top1_acc, top5_acc = accuracy(student_logits, labels, topk = (1, 5))

        with torch.no_grad():
            if config["e_size"] == 7:
                # with torch.cuda.amp.autocast():
                teacher_logits, teacher_maps = teacher(maps, out_maps = True)

                loss_maps["student_maps"] = student_maps['size_7']
                loss_maps["teacher_maps"] = teacher_maps
            else:
                teacher_logits = teacher(images)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = criterion(student_logits, teacher_logits, labels, **loss_maps) / config["accumulation"]
            scaler.scale(loss).backward()
        else:
            loss = criterion(student_logits, teacher_logits, labels, **loss_maps) / config["accumulation"]
            loss.backward()

        if ((batch + 1) % config["accumulation"] == 0) or (batch + 1 == len(loader)):
            if scaler is not None:
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
            else:
                optimizer.step()
                optimizer.zero_grad()

            if scheduler is not None: scheduler.step()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(student_logits.detach().cpu().numpy())

        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
        
    epoch_outputs = torch.as_tensor(np.array(outputs_), dtype = torch.float32)
    epoch_labels  = torch.as_tensor(np.array(labels_))

    epoch_top1_acc, epoch_top5_acc = accuracy(epoch_outputs, epoch_labels, topk = (1, 5))
    
    
    if USE_WANDB: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()


def train_baseline(model, loaders, optimizer, scheduler, criterion, epoch, config):
    model.train()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    (CIFAR100, TinyImageNet, ImageNetSketch) = loaders

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (CIFAR100_Item, TinyImageNet_Item, ImageNetSketch_Item) in enumerate(zip(cycle(CIFAR100), TinyImageNet, cycle(ImageNetSketch))):
        images, labels = CIFAR100_Item
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images, head = "CIFAR100")
        loss    = criterion(outputs, labels) * 0.33

        if config["dataset"] == "CIFAR100":
            top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))
            labels_.extend(labels.detach().cpu().numpy())
            outputs_.extend(outputs.detach().cpu().numpy())

        images, labels = TinyImageNet_Item
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs  = model(images, head = "TinyImageNet")
        loss    += criterion(outputs, labels) * 0.33

        if config["dataset"] == "TinyImageNet":
            top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))
            labels_.extend(labels.detach().cpu().numpy())
            outputs_.extend(outputs.detach().cpu().numpy())

        images, labels = ImageNetSketch_Item
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs  = model(images, head = "ImageNetSketch")
        loss    += criterion(outputs, labels) * 0.33

        if config["dataset"] == "ImageNetSketch":
            top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))
            labels_.extend(labels.detach().cpu().numpy())
            outputs_.extend(outputs.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(TinyImageNet):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(TinyImageNet)}], " + \
                      f"{time_since(start, float(batch + 1) / len(TinyImageNet))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
    
    epoch_outputs = torch.as_tensor(np.array(outputs_), dtype = torch.float32)
    epoch_labels  = torch.as_tensor(np.array(labels_))

    epoch_top1_acc, epoch_top5_acc = accuracy(epoch_outputs, epoch_labels, topk = (1, 5))

    if USE_WANDB: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()


def train_baseline_extended(model, loaders, optimizer, scheduler, criterion, epoch, config):
    model.train()

    losses    = AverageMeter()
    top1_accs = AverageMeter()
    top5_accs = AverageMeter()

    (CIFAR100, TinyImageNet, ImageNetSketch) = loaders

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (CIFAR100_Item, TinyImageNet_Item, ImageNetSketch_Item) in enumerate(zip(cycle(CIFAR100), TinyImageNet, cycle(ImageNetSketch))):
        CIFAR100_Images, CIFAR100_Labels             = CIFAR100_Item
        TinyImageNet_Images, TinyImageNet_Labels     = TinyImageNet_Item
        ImageNetSketch_Images, ImageNetSketch_Labels = ImageNetSketch_Item

        TinyImageNet_Labels   = TinyImageNet_Labels + 100
        ImageNetSketch_Labels = ImageNetSketch_Labels + 300

        images = torch.cat((CIFAR100_Images, TinyImageNet_Images, ImageNetSketch_Images), 0)
        labels = torch.cat((CIFAR100_Labels, TinyImageNet_Labels, ImageNetSketch_Labels), 0)

        indices = torch.randperm(images.shape[0])
        images, labels = images[indices], labels[indices]
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        top1_acc, top5_acc = accuracy(outputs, labels, topk = (1, 5))
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        optimizer.zero_grad()

        losses.update(RD(loss.item()), images.shape[0])
        top1_accs.update(RD(top1_acc.item()), images.shape[0])
        top5_accs.update(RD(top5_acc.item()), images.shape[0])
        
        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_acc@1": top1_accs.average, "train_avg_acc@5" : top5_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(TinyImageNet):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(TinyImageNet)}], " + \
                      f"{time_since(start, float(batch + 1) / len(TinyImageNet))}, " + \
                      f"Acc@1: {top1_accs.value:.3f}({top1_accs.average:.3f}), " + \
                      f"Acc@5: {top5_accs.value:.3f}({top5_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)
    
    epoch_outputs = torch.as_tensor(np.array(outputs_), dtype = torch.float32)
    epoch_labels  = torch.as_tensor(np.array(labels_))

    epoch_top1_acc, epoch_top5_acc = accuracy(epoch_outputs, epoch_labels, topk = (1, 5))

    if USE_WANDB: wandb.log({"train_epoch_acc@1": epoch_top1_acc, "train_epoch_acc@5" : epoch_top5_acc})
    free_gpu_memory(DEVICE)
    return epoch_top1_acc.item()


def patch_level_accuracy(outputs, labels, num_patches = 9):
    outputs = outputs.reshape((-1, num_patches, num_patches))

    patch_accuracy = torch.zeros([1], device = DEVICE)
    # print("******************* Inside Patch Acc *******************")
    # print("outputs: ")
    # print(outputs)
    # print("targets: ")
    # print(labels)
    for patch_idx in range(outputs.shape[1]):
        logits = outputs[:, patch_idx, :]
        label  = labels[:, patch_idx]

        # print("Logits: ", logits)
        # print("Label: ", label)
        _, predicted = torch.max(logits, dim = 1)
        # print("Predicted: ", predicted)
        correct_predictions = (predicted == label).float().mean()

        patch_accuracy = patch_accuracy + (correct_predictions / num_patches)
    
    # print("******************* Out Patch Acc *******************")
    return patch_accuracy

def image_level_accuracy(outputs, labels, num_patches = 9):
    batch_size = outputs.shape[0]
    outputs = outputs.reshape((-1, num_patches, num_patches))

    boolean_predictions = torch.zeros([batch_size, num_patches], device = DEVICE)
    for patch_idx in range(outputs.shape[1]):
        logits   = outputs[:, patch_idx, :]
        position = labels[:, patch_idx]

        _, predicted = torch.max(logits, dim = 1)
        correct_predictions = (predicted == position)

        boolean_predictions[:, patch_idx] = correct_predictions
    
    image_accuracy = (boolean_predictions.sum(dim = 1) == num_patches).float().mean()
    return image_accuracy

def imshow_two_images(inp1, inp2, title=None, means=IMAGENET_MEANS, stds=IMAGENET_STDS):
    # Convert images from tensor to numpy and denormalize
    inp1 = inp1.numpy().transpose((1, 2, 0))
    inp2 = inp2.numpy().transpose((1, 2, 0))
    mean = np.array(means)
    std  = np.array(stds)
    inp1 = std * inp1 + mean
    inp2 = std * inp2 + mean
    inp1 = np.clip(inp1, 0, 1)
    inp2 = np.clip(inp2, 0, 1)
    
    # Combine images horizontally
    combined_image = np.hstack((inp1[:, :, [2, 1, 0]], inp2[:, :, [2, 1, 0]]))
    
    # Display combined image
    plt.imshow(combined_image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def train_jigsaw(model, loader, optimizer, scheduler, criterion, epoch, accumulation_steps, scaler, grid_size):
    model.train()

    losses     = AverageMeter()
    patch_accs = AverageMeter()
    image_accs = AverageMeter()

    labels_, outputs_ = [], []
    start = end = time.time()
    for batch, (images, labels) in enumerate(loader):

        # batch, (images, shuffled_image, labels) = 0, next(iter(loader))
        # print("Image shape: ", images.shape)
        # print("Labels: ", labels)
        # imshow_two_images(images.squeeze(0), shuffled_image.squeeze(0))
        # imshow(images.squeeze(0))
        # imshow(shuffled_image.squeeze(0))

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels) / accumulation_steps

        # print(loss)
        scaler.scale(loss).backward()

        patch_accuracy = patch_level_accuracy(outputs, labels, num_patches = grid_size * grid_size)
        image_accuracy = image_level_accuracy(outputs, labels, num_patches = grid_size * grid_size)
        # print(f"pAcc: {patch_accuracy}, iAcc: {image_accuracy}")
        
        # print(outputs.shape)
        # outputs = outputs.reshape((-1, 4, 4))
        # values, predicted = torch.max(outputs, dim = 2)
        # print(outputs)
        # print("values: ", values)
        # print("predictions: ", predicted)
        # print("labels: ", labels)

        if ((batch + 1) % accumulation_steps == 0) or (batch + 1 == len(loader)):
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

            if scheduler is not None: scheduler.step()

        losses.update(RD(loss.item()), images.shape[0])
        patch_accs.update(RD(patch_accuracy.item()), images.shape[0])
        image_accs.update(RD(image_accuracy.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

        end = time.time()
        if USE_WANDB: 
            wandb.log({"train_avg_pacc": patch_accs.average, "train_avg_iacc" : image_accs.average, \
                        "train_avg_loss": losses.average, "lr": optimizer.param_groups[0]['lr']})

        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[T] E/B: [{epoch + 1}][{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"PAcc: {patch_accs.value:.3f}({patch_accs.average:.3f}), " + \
                      f"IAcc: {image_accs.value:.3f}({image_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f}), LR: {optimizer.param_groups[0]['lr']:.6f}"
            
            print(message)

        # return

    epoch_outputs = torch.as_tensor(np.array(outputs_), dtype = torch.float32)
    epoch_labels  = torch.as_tensor(np.array(labels_))

    patch_epoch_accuracy = patch_level_accuracy(epoch_outputs, epoch_labels, num_patches = grid_size * grid_size)
    image_epoch_accuracy = image_level_accuracy(epoch_outputs, epoch_labels, num_patches = grid_size * grid_size)

    if USE_WANDB: wandb.log({"train_epoch_pacc": patch_epoch_accuracy, "train_epoch_iacc" : image_epoch_accuracy})
    free_gpu_memory(DEVICE)
    return image_epoch_accuracy.item()


def validate_jigsaw(model, loader, criterion, config, grid_size):
    model.eval()

    losses     = AverageMeter()
    patch_accs = AverageMeter()
    image_accs = AverageMeter()

    labels_, outputs_ = [], []
    start  = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)

        patch_accuracy = patch_level_accuracy(outputs, labels, num_patches = grid_size * grid_size)
        image_accuracy = image_level_accuracy(outputs, labels, num_patches = grid_size * grid_size)
        loss = criterion(outputs, labels)

        losses.update(RD(loss.item()), images.shape[0])
        patch_accs.update(RD(patch_accuracy.item()), images.shape[0])
        image_accs.update(RD(image_accuracy.item()), images.shape[0])
        
        labels_.extend(labels.detach().cpu().numpy())
        outputs_.extend(outputs.detach().cpu().numpy())

        end = time.time()
        if USE_WANDB: wandb.log({"valid_avg_pacc": patch_accs.average, "valid_avg_iacc" : image_accs.average, "valid_avg_loss": losses.average})
        if (batch + 1) % PRINT_FREQ == 0 or (batch + 1) == len(loader):
            message = f"[V] B: [{batch + 1}/{len(loader)}], " + \
                      f"{time_since(start, float(batch + 1) / len(loader))}, " + \
                      f"PAcc: {patch_accs.value:.3f}({patch_accs.average:.3f}), " + \
                      f"IAcc: {image_accs.value:.3f}({image_accs.average:.3f}), " + \
                      f"Loss: {losses.value:.3f}({losses.average:.3f})"

            print(message)

    epoch_outputs = torch.as_tensor(np.array(outputs_), dtype = torch.float32)
    epoch_labels  = torch.as_tensor(np.array(labels_))

    patch_epoch_accuracy = patch_level_accuracy(epoch_outputs, epoch_labels, num_patches = grid_size * grid_size)
    image_epoch_accuracy = image_level_accuracy(epoch_outputs, epoch_labels, num_patches = grid_size * grid_size)

    if USE_WANDB: wandb.log({"valid_epoch_pacc": patch_epoch_accuracy, "valid_epoch_iacc" : image_epoch_accuracy})
    free_gpu_memory(DEVICE)
    return image_epoch_accuracy.item()
