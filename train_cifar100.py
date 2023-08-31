from common      import *
from models      import *
from dataset     import *
from labels      import *
from config_file import CFG

def run():
    seed_everything(SEED)

    if CFG["use_wandb"]: 
        wandb.init(project = "CIFAR100-Training", config = CFG)

    train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
    test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)

    train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(CFG["image_size"]),
        ] + CFG['train_transforms'] + [
        T.ToTensor(),
        T.Normalize(CIFAR_MEANS, CIFAR_STDS)
    ])

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(CFG["image_size"]),
        ] + CFG['test_transforms'] + [
        T.ToTensor(),
        T.Normalize(CIFAR_MEANS, CIFAR_STDS)
    ])

    trainset = CIFAR100Dataset(train_images, train_labels, train_transforms)
    testset  = CIFAR100Dataset(test_images, test_labels, test_transforms)

    trainloader = DataLoader(trainset, **CFG["trainloader"])
    testloader  = DataLoader(testset, **CFG["testloader"])

    model = BaselineModel(CFG["model"])
    model.to(DEVICE)
    
    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = nn.CrossEntropyLoss()

    best_model, best_accuracy = None, 0
    if CFG["use_wandb"]: wandb.watch(model, criterion, log = "all", log_freq = 10)
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train(model, trainloader, optimizer, scheduler, criterion, epoch)
        valid_top1_acc = validate(model, testloader, criterion)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(model)
            best_accuracy = valid_top1_acc

        if train_top1_acc - valid_top1_acc > 10.0:
            print("Early stopping condition...")
            break

    if CFG["use_wandb"]: 
        torch.save(model.state_dict(), f"./weights/cifar100/{wandb.run.name}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()