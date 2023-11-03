from common        import *
from dataset       import *
from labels        import *
from models        import *
from config_file   import *
from procedures    import train, validate

def run():
    CFG = config
    seed_everything(SEED)

    train_data_c = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/train_{CFG['dataset']}_dataset_CIFAR100_model_512x7x7.npy"       , mmap_mode = "r+")
    train_data_t = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/train_{CFG['dataset']}_dataset_TinyImageNet_model_1280x7x7.npy"  , mmap_mode = "r+")
    train_data_s = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/train_{CFG['dataset']}_dataset_ImageNetSketch_model_2048x7x7.npy", mmap_mode = "r+")
    train_labels = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/train_{CFG['dataset']}_labels.npy"                               , mmap_mode = "r+")

    test_data_c = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_dataset_CIFAR100_model_512x7x7.npy"        , mmap_mode = "r+")
    test_data_t = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_dataset_TinyImageNet_model_1280x7x7.npy"   , mmap_mode = "r+")
    test_data_s = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_dataset_ImageNetSketch_model_2048x7x7.npy" , mmap_mode = "r+")
    test_labels = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_labels.npy"                                , mmap_mode = "r+")

    trainset = TeacherDataset(train_data_c, train_data_t, train_data_s, train_labels)
    testset  = TeacherDataset(test_data_c,  test_data_t,  test_data_s,  test_labels)

    trainloader = DataLoader(trainset, **CFG["trainloader"])
    testloader  = DataLoader(testset, **CFG["testloader"])

    model = ConvTeacherModel(CFG["conv_teacher"])
    model.to(DEVICE)

    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = nn.CrossEntropyLoss()

    if USE_WANDB: 
        wandb.init(project = f"{CFG['dataset']}-Training", name = RUN_NAME, config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train(model, trainloader, optimizer, scheduler, criterion, epoch)
        valid_top1_acc = validate(model, testloader, criterion)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(model)
            best_accuracy = np.round(valid_top1_acc, 3)
            best_epoch    = epoch

        if train_top1_acc - valid_top1_acc > 10.0:
            print("Early stopping condition...")
            break
    
    if USE_WANDB: 
        torch.save(best_model.state_dict(), f"./weights/teachers/stage-2/{CFG['dataset']}/{RUN_NAME}_epoch_{best_epoch}_acc@1_{best_accuracy}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()