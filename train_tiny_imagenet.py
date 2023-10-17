from common      import *
from models      import *
from dataset     import *
from labels      import *
from config_file import *
from procedures  import train, validate

def run():
    CFG = student_config
    seed_everything(SEED)

    if USE_WANDB: 
        wandb.init(project = "TinyImageNet-Training", config = CFG)

    train_df = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
    test_df  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)

    train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(CFG["image_size"]),
        ] + CFG['train_transforms'] + [
        T.ToTensor(),
        T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(CFG["image_size"]),
        ] + CFG['test_transforms'] + [
        T.ToTensor(),
        T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    trainset = ImageNetDataset(train_df, train_transforms)
    testset  = ImageNetDataset(test_df, test_transforms)

    trainloader = DataLoader(trainset, **CFG["trainloader"])
    testloader  = DataLoader(testset, **CFG["testloader"])

    model = IntermediateModel(CFG["model"], CFG["n_embedding"], CFG["activation"])
    model.to(DEVICE)
    
    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = nn.CrossEntropyLoss()

    best_model, best_accuracy, best_epoch = None, 0, None
    if USE_WANDB: wandb.watch(model, criterion, log = "all", log_freq = 10)
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train(model, trainloader, optimizer, scheduler, criterion, epoch)
        valid_top1_acc = validate(model, testloader, criterion)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(model)
            best_accuracy = np.round(valid_top1_acc, 2)
            best_epoch    = epoch

        if train_top1_acc - valid_top1_acc > 10.0:
            print("Early stopping condition...")
            break

    if USE_WANDB: 
        torch.save(best_model.state_dict(), f"./weights/students/tiny-imagenet/{wandb.run.name}_epoch_{best_epoch}_acc@1_{best_accuracy}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()