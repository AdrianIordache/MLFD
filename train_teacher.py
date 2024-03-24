from common        import *
from dataset       import *
from labels        import *
from models        import *
from config_file   import *
from procedures    import train, validate

def run():
    CFG = config
    seed_everything(SEED)

    PATH_TO_TRAIN_EMBEDDINGS = f"./embeddings/stage-4/part-3/{CFG['dataset']}/train/"
    PATH_TO_TEST_EMBEDDINGS  = f"./embeddings/stage-4/part-3/{CFG['dataset']}/test/"

    PATH_TO_TRAIN_SAMPLES = os.path.join(PATH_TO_TRAIN_EMBEDDINGS, f"{CFG['e_size']}x{CFG['e_size']}")
    PATH_TO_TEST_SAMPLES  = os.path.join(PATH_TO_TEST_EMBEDDINGS,  f"{CFG['e_size']}x{CFG['e_size']}")

    PATH_TO_TRAIN_LABELS  = os.path.join(PATH_TO_TRAIN_EMBEDDINGS, f"{CFG['dataset']}_train_labels.npy")
    PATH_TO_TEST_LABELS   = os.path.join(PATH_TO_TEST_EMBEDDINGS,  f"{CFG['dataset']}_test_labels.npy")

    trainset = TeacherDataset(PATH_TO_TRAIN_SAMPLES, PATH_TO_TRAIN_LABELS, **CFG["n_features_maps"])
    testset  = TeacherDataset(PATH_TO_TEST_SAMPLES, PATH_TO_TEST_LABELS, **CFG["n_features_maps"])

    trainloader = DataLoader(trainset, **CFG["trainloader"])
    testloader  = DataLoader(testset, **CFG["testloader"])

    model = S7Teacher({**CFG["teacher"], **CFG["n_features_maps"]})
    model.to(DEVICE)

    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = nn.CrossEntropyLoss(label_smoothing = CFG["label_smoothing"])
    scaler    = torch.cuda.amp.GradScaler()

    # image, _ = next(iter(trainloader))
    # model(image.to(DEVICE))

    if USE_WANDB: 
        wandb.init(project = f"{CFG['dataset']}-Training", name = RUN_NAME, config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train(model, trainloader, optimizer, scheduler, criterion, epoch, None, CFG["accumulation"], scaler)
        valid_top1_acc = validate(model, testloader, criterion, CFG)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(model)
            best_accuracy = np.round(valid_top1_acc, 3)
            best_epoch    = epoch

    if USE_WANDB:     
        PATH_TO_SAVED_MODEL = f"./weights/teachers/stage-4/{CFG['dataset']}"
        os.makedirs(PATH_TO_SAVED_MODEL, exist_ok = True)
        torch.save(best_model.state_dict(), f"{PATH_TO_SAVED_MODEL}/{RUN_NAME}_epoch_{best_epoch}_acc@1_{best_accuracy}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()