from common      import *
from models      import *
from dataset     import *
from labels      import *
from config_file import *
from procedures  import train_baseline, train_baseline_extended, validate

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def run():
    CFG = config
    print(f"CFG: {CFG}")
    seed_everything(SEED)

    trainloaders, testloader = get_dataloaders_baseline(CFG)

    model = BaselineExtendedModel(CFG['expert'])
    model.to(DEVICE)
    
    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = nn.CrossEntropyLoss()

    if USE_WANDB: 
        wandb.init(project = f"{CFG['dataset']}-Training", name = RUN_NAME, config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train_baseline_extended(model, trainloaders, optimizer, scheduler, criterion, epoch, CFG)
        valid_top1_acc = validate(model, testloader, criterion, CFG)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(model)
            best_accuracy = np.round(valid_top1_acc, 3)
            best_epoch    = epoch

        if train_top1_acc - valid_top1_acc > 10.0:
            print("Early stopping condition...")
            break

    if USE_WANDB: 
        torch.save(best_model.state_dict(), f"./weights/experts/stage-3/{CFG['dataset']}/{RUN_NAME}_epoch_{best_epoch}_acc@1_{best_accuracy}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()