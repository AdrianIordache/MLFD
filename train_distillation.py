from common        import *
from dataset       import *
from labels        import *
from models        import *
from config_file   import *
from procedures    import train, validate, train_kd
from preprocessing import load_cifar

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, distillation_loss, temperature = 5, kd_alpha = 0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.distillation_loss = distillation_loss
        self.temperature       = temperature
        self.kd_alpha          = kd_alpha

    def forward(self, student_logits, teacher_logits, labels):
        softmax       = nn.Softmax(dim = 1)
        log_softmax   = nn.LogSoftmax(dim = 1)
        cross_entropy = nn.CrossEntropyLoss()

        teacher_probs = softmax(teacher_logits / self.temperature)
        student_probs = log_softmax(student_logits / self.temperature)

        dist_loss = self.distillation_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        ce_loss   = cross_entropy(student_logits, labels)
        
        return self.kd_alpha * dist_loss + (1.0 - self.kd_alpha) * ce_loss

def run():
    CFG = student_config
    seed_everything(SEED)

    PATH_TO_TRAIN_EMBEDDINGS = f"./embeddings/{teacher_config['dataset']}_train_embeddings.csv"
    PATH_TO_TEST_EMBEDDINGS  = f"./embeddings/{teacher_config['dataset']}_train_embeddings.csv"

    teacher = nn.Sequential(OrderedDict([
            ('dropout',    nn.Dropout(p = teacher_config["p_dropout"])),
            ('projection', nn.Linear(6144, teacher_config["n_outputs"]))
        ])
    )

    teacher.load_state_dict(torch.load(f"/usr/app/weights/teachers/{teacher_config['dataset']}/earnest-jazz-10_epoch_26_acc@1_0.47.pt"))
    teacher.to(DEVICE)
    teacher.eval()

    if teacher_config['dataset'] == "CIFAR100":
        means, stds = CIFAR_MEANS, CIFAR_STDS
    else:
        means, stds = IMAGENET_MEANS, IMAGENET_STDS

    train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((CFG["image_size"], CFG["image_size"])),
        ] + CFG['train_transforms'] + [
        T.ToTensor(),
        T.Normalize(means, stds)
    ])

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((CFG["image_size"], CFG["image_size"])),
        ] + CFG['test_transforms'] + [
        T.ToTensor(),
        T.Normalize(means, stds)
    ])

    trainset, testset = None, None

    if teacher_config['dataset'] == "CIFAR100":
        train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
        test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)

        trainset = CIFAR100Dataset(train_images, train_labels, train_transforms, PATH_TO_TRAIN_EMBEDDINGS)
        testset  = CIFAR100Dataset(test_images, test_labels, test_transforms)

    if teacher_config['dataset'] == "TinyImageNet":
        train_tiny_imagenet = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
        test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
        
        trainset = ImageNetDataset(train_tiny_imagenet, train_transforms, PATH_TO_TRAIN_EMBEDDINGS)
        testset  = ImageNetDataset(test_tiny_imagenet, test_transforms)
        
    if teacher_config['dataset'] == "ImageNetSketch":
        train_imagenet_sketch = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
        test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
        
        trainset = ImageNetDataset(train_imagenet_sketch, train_transforms, PATH_TO_TRAIN_EMBEDDINGS)
        testset  = ImageNetDataset(test_imagenet_sketch, test_transforms)
        
    trainloader = DataLoader(trainset, **CFG["trainloader"])
    testloader  = DataLoader(testset, **CFG["testloader"])

    student = IntermediateModel(CFG["model"], CFG["n_embedding"], CFG["activation"])
    student.to(DEVICE)

    optimizer    = get_optimizer(student.parameters(), CFG)
    scheduler    = get_scheduler(optimizer, CFG)
    criterion    = nn.CrossEntropyLoss()
    criterion_kd = KnowledgeDistillationLoss(
        distillation_loss = CFG["distillation_loss"],
        temperature       = CFG["temperature"],
        kd_alpha          = CFG["kd_alpha"]
    )

    if USE_WANDB: 
        wandb.init(project = f"{teacher_config['dataset']}-Training", config = CFG)

    best_model, best_accuracy, best_epoch = None, 0, None
    for epoch in range(CFG["epochs"]):
        train_top1_acc = train_kd(student, teacher, trainloader, optimizer, scheduler, criterion_kd, epoch)
        valid_top1_acc = validate(student, testloader, criterion)

        if valid_top1_acc > best_accuracy:
            best_model    = copy.deepcopy(student)
            best_accuracy = valid_top1_acc
            best_epoch    = epoch

    if USE_WANDB: 
        torch.save(best_model.state_dict(), f"./weights/experts/{teacher_config['dataset']}/{wandb.run.name}_epoch_{best_epoch}_acc@1_{np.round(best_accuracy, 2)}.pt")
        wandb.finish()

if __name__ == "__main__":
    run()