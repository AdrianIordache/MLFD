from common        import *
from dataset       import *
from labels        import *
from models        import *
from config_file   import *
from procedures    import train, validate, train_kd_conv

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, distillation_loss, feature_map_loss, temperature = 5, kd_coefs = 0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.distillation_loss = distillation_loss
        self.feature_map_loss  = feature_map_loss
        self.temperature       = temperature
        self.kd_coefs          = kd_coefs

    def forward(self, student_logits, teacher_logits, student_maps, teacher_maps, labels):
        softmax       = nn.Softmax(dim = 1)
        log_softmax   = nn.LogSoftmax(dim = 1)
        cross_entropy = nn.CrossEntropyLoss()

        teacher_probs = softmax(teacher_logits / self.temperature)
        student_probs = log_softmax(student_logits / self.temperature)

        maps_loss = self.feature_map_loss(student_maps, teacher_maps)
        dist_loss = self.distillation_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        ce_loss   = cross_entropy(student_logits, labels)
        
        return self.kd_coefs[0] * dist_loss + self.kd_coefs[1] * maps_loss + self.kd_coefs[2] * ce_loss

def run():
    CFG = config
    seed_everything(SEED)

    teacher = ConvTeacherModel(CFG["conv_teacher"])
    teacher.load_state_dict(torch.load(f"./weights/teachers/stage-2/{CFG['dataset']}/exp-14-conv-teacher_epoch_964_acc@1_0.615.pt"))
    teacher.to(DEVICE)
    teacher.eval()

    trainloader, testloader = get_dataloaders(CFG, distillation = True)

    student = IntermediateModel(CFG["expert"], CFG["n_embedding"], CFG["activation"])
    student.to(DEVICE)

    optimizer    = get_optimizer(student.parameters(), CFG)
    scheduler    = get_scheduler(optimizer, CFG)
    criterion    = nn.CrossEntropyLoss()

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
        train_top1_acc = train_kd_conv(student, teacher, trainloader, optimizer, scheduler, criterion_kd, epoch)
        valid_top1_acc = validate(student, testloader, criterion)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(student)
            best_accuracy = valid_top1_acc
            best_epoch    = epoch

    if USE_WANDB: 
        torch.save(best_model.state_dict(), f"./weights/students/stage-2/{CFG['dataset']}/{RUN_NAME}_epoch_{best_epoch}_acc@1_{np.round(best_accuracy, 3)}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()