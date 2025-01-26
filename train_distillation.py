from common        import *
from dataset       import *
from labels        import *
from models        import *
from config_file   import *
from procedures    import train, validate, train_kd_conv, train_kd

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, distillation_loss, feature_map_loss, temperature = 5, kd_coefs = 0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.distillation_loss = distillation_loss
        self.feature_map_loss  = feature_map_loss
        self.temperature       = temperature
        self.kd_coefs          = kd_coefs

    def forward(self, student_logits, teacher_logits, labels, student_maps = None, teacher_maps = None):
        softmax       = nn.Softmax(dim = 1)
        log_softmax   = nn.LogSoftmax(dim = 1)
        cross_entropy = nn.CrossEntropyLoss()

        teacher_probs = softmax(teacher_logits / self.temperature)
        student_probs = log_softmax(student_logits / self.temperature)

        dist_loss = self.distillation_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        ce_loss   = cross_entropy(student_logits, labels)

        if student_maps is not None and teacher_maps is not None:
            maps_loss = self.feature_map_loss(student_maps, teacher_maps)
            return self.kd_coefs[0] * dist_loss + self.kd_coefs[1] * maps_loss + self.kd_coefs[2] * ce_loss
        
        return self.kd_coefs[0] * dist_loss + self.kd_coefs[1] * ce_loss

# student_maps = torch.permute(student_maps, (0, 3, 1, 2))
def run():
    print(config)
    CFG = config
    seed_everything(SEED)

    # DEVICE = torch.device('cpu')
    teacher = S7Teacher({**CFG["teacher"], **CFG["n_features_maps"]})

    # f"./weights/teachers/stage-3/CIFAR100/exp-19-teacher-8_epoch_202_acc@1_0.715.pt"
    # f"./weights/teachers/stage-3/CIFAR100/exp-18-teacher-1_epoch_421_acc@1_0.678.pt
    # f"./weights/teachers/stage-3/TinyImageNet/exp-28-teacher-1_epoch_462_acc@1_0.642.pt"
    # f"./weights/teachers/stage-3/TinyImageNet/exp-29-teacher-8_epoch_126_acc@1_0.655.pt"
    # f"./weights/teachers/stage-3/ImageNetSketch/exp-16-teacher-1_epoch_1657_acc@1_0.684.pt" # exp-4-teacher-p3_epoch_251_acc@1_0.573.pt #exp-4-teacher-p3_epoch_251_acc@1_0.573.pt
    teacher.load_state_dict(torch.load(f"./weights/teachers/stage-6/OxfordPets/exp-6-teacher-single_epoch_140_acc@1_0.651.pt")) #f"./weights/teachers/stage-4/{CFG['dataset']}/exp-4-teacher-blind_epoch_237_acc@1_0.542.pt"))
    teacher.to(DEVICE)
    teacher.eval()

    # teacher = ExpertModel(CFG["expert_t"], CFG["n_embedding"], CFG["activation"])
    # teacher.load_state_dict(torch.load(f"./weights/experts/stage-1/{CFG['dataset']}/exp-34-larger-experts_epoch_194_acc@1_0.644.pt"))
    # teacher.to(DEVICE)
    # teacher.eval()

    trainloader, testloader = get_dataloaders_advanced(CFG, distillation = True)
    mixup_fn = None if CFG["use_mixup"] == False else Mixup(**CFG['mixup_param'])

    student = ExpertModel(CFG["expert"], CFG["n_embedding"], CFG["activation"])
    student.to(DEVICE)
    # print(student)

    optimizer   = get_optimizer(student.parameters(), CFG)
    scheduler   = get_scheduler(optimizer, CFG)
    v_criterion = nn.CrossEntropyLoss(label_smoothing = CFG["label_smoothing"])
    scaler      = torch.cuda.amp.GradScaler() if CFG["mixed_precision"] else None

    criterion_kd = KnowledgeDistillationLoss(
        distillation_loss = CFG["distillation_loss"],
        feature_map_loss  = CFG["feature_map_loss"],
        temperature       = CFG["temperature"],
        kd_coefs          = CFG["kd_coefs"]
    )

    if USE_WANDB: 
        wandb.init(project = f"{CFG['dataset']}-Training", name = RUN_NAME, config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        tic = time.time()
        train_top1_acc = train_kd(student, teacher, trainloader, optimizer, scheduler, criterion_kd, epoch, scaler, CFG)
        valid_top1_acc = validate(student, testloader, v_criterion, CFG)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(student)
            best_accuracy = valid_top1_acc
            best_epoch    = epoch

        toc = time.time()
        print(f"Epoch time: {(toc - tic)}'s")

    if USE_WANDB: 
        PATH_TO_SAVED_MODEL = f"./weights/students/stage-5/{CFG['dataset']}/"
        os.makedirs(PATH_TO_SAVED_MODEL, exist_ok = True)
        torch.save(best_model.state_dict(), f"{PATH_TO_SAVED_MODEL}/{RUN_NAME}_epoch_{best_epoch}_acc@1_{best_accuracy}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()