from common        import *
from dataset       import *
from labels        import *
from models        import *
from config_file   import *
from procedures    import train, validate

def run():
    CFG = teacher_config
    seed_everything(SEED)

    train_df = pol.read_csv(f"./embeddings/{CFG['dataset']}_train_embeddings.csv")
    test_df  = pol.read_csv(f"./embeddings/{CFG['dataset']}_test_embeddings.csv")

    features = [f"cifar_x{i}" for i in range(0, 2048)] + \
               [f"tiny_imagenet_x{i}" for i in range(0, 2048)] + \
               [f"imagenet_sketch_x{i}" for i in range(0, 2048)]

    X_train = train_df[features].to_numpy()
    X_test  =  test_df[features].to_numpy()

    y_train = train_df["label"].to_numpy()
    y_test  =  test_df["label"].to_numpy()

    X_train = torch.tensor(X_train, dtype = torch.float32)
    y_train = torch.tensor(y_train, dtype = torch.long)

    X_test  = torch.tensor(X_test, dtype = torch.float32)
    y_test  = torch.tensor(y_test, dtype = torch.long)

    trainset = TensorDataset(X_train, y_train)
    testset  = TensorDataset(X_test,  y_test)

    trainloader = DataLoader(trainset, **CFG["trainloader"])
    testloader  = DataLoader(testset, **CFG["testloader"])

    model = nn.Sequential(OrderedDict([
            ('dropout',    nn.Dropout(p = CFG["p_dropout"])),
            ('projection', nn.Linear(X_train.shape[1], CFG["n_outputs"]))
        ])
    )
    CFG["model"] = model
    model.to(DEVICE)

    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = nn.CrossEntropyLoss()

    if USE_WANDB: 
        wandb.init(project = f"{CFG['dataset']}-Training", config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train(model, trainloader, optimizer, scheduler, criterion, epoch)
        valid_top1_acc = validate(model, testloader, criterion)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(model)
            best_accuracy = np.round(valid_top1_acc, 2)
            best_epoch    = epoch

    if USE_WANDB: 
        torch.save(best_model.state_dict(), f"./weights/teachers/{CFG['dataset']}/{wandb.run.name}_epoch_{best_epoch}_acc@1_{np.round(best_accuracy, 2)}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()