from common      import *
from models      import *
from dataset     import *
from labels      import *
from config_file import *
from procedures  import train, validate

from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

def run():
    CFG = config
    print(f"CFG: {CFG}")
    seed_everything(SEED)
    
    trainloader, testloader = get_dataloaders(CFG)
    mixup_fn = None if CFG["use_mixup"] == False else Mixup(**CFG['mixup_param'])
    
    model = ExpertModel(CFG["expert"], CFG["n_embedding"], CFG["activation"])
    model.to(DEVICE)
    
    optimizer   = get_optimizer(model.parameters(), CFG)
    scheduler   = get_scheduler(optimizer, CFG)
    t_criterion = SoftTargetCrossEntropy() if CFG["use_mixup"] == True else nn.CrossEntropyLoss(label_smoothing = CFG["label_smoothing"])
    v_criterion = nn.CrossEntropyLoss(label_smoothing = CFG["label_smoothing"])

    if USE_WANDB: 
        wandb.init(project = f"{CFG['dataset']}-Advanced", name = RUN_NAME, config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train(model, trainloader, optimizer, scheduler, t_criterion, epoch, mixup_fn)
        valid_top1_acc = validate(model, testloader, v_criterion)

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