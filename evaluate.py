from common        import *
from models        import *
from dataset       import *
from procedures    import evaluate
from preprocessing import load_cifar

CFG = dict(
    CIFAR100 = dict(
        model_name  = "resnet18",
        num_classes = 100,
        pretrained  = False,
    ),

    TinyImageNet = dict(
        model_name  = "tf_efficientnet_b0",
        num_classes = 200,
        pretrained  = False,
    ),

    ImageNetSketch = dict(
        model_name  = "seresnext26t_32x4d",
        num_classes = 1000,
        pretrained  = False,
    ),

    loader = dict(
        batch_size     = 1024,
        shuffle        = False, 
        num_workers    = 2,
        pin_memory     = True,
        sampler        = sampler, 
        worker_init_fn = seed_worker, 
        drop_last      = False
    ),

    n_embedding = 2048,
    activation  = nn.Identity(),
    image_size = 224,
    print_freq = 10,

    dataset    = "ImageNetSketch",

    conv_teacher = dict(
        nf_space   = 4096,
        nf_outputs = 2048,
        n_outputs  = 1000,

        nf_c = 512,
        nf_t = 1280,
        nf_s = 2048
    ),
)

def run(path_to_model, data_type):
    assert data_type in ["CIFAR100", "TinyImageNet", "ImageNetSketch"]

    model = IntermediateModel(CFG[data_type], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    if data_type == "CIFAR100":
        means, stds = CIFAR_MEANS, CIFAR_STDS
    else:
        means, stds = IMAGENET_MEANS, IMAGENET_STDS

    criterion  = nn.CrossEntropyLoss()
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((CFG["image_size"], CFG["image_size"])),
        T.ToTensor(),
        T.Normalize(means, stds) 
    ])

    trainset, testset = None, None

    if data_type == "CIFAR100":
        train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
        test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)

        trainset = CIFAR100Dataset(train_images, train_labels, transforms)
        testset  = CIFAR100Dataset(test_images, test_labels, transforms)

    if data_type == "TinyImageNet":
        train_tiny_imagenet = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
        test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
        
        trainset = ImageNetDataset(train_tiny_imagenet, transforms)
        testset  = ImageNetDataset(test_tiny_imagenet, transforms)
        
    if data_type == "ImageNetSketch":
        train_imagenet_sketch = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
        test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
        
        trainset = ImageNetDataset(train_imagenet_sketch, transforms)
        testset  = ImageNetDataset(test_imagenet_sketch, transforms)
        
    trainloader = DataLoader(trainset, **CFG["loader"])
    testloader  = DataLoader(testset, **CFG["loader"])

    top1_acc, top5_acc = evaluate(model, testloader, criterion)
    print(f"[E] Acc@1: {np.round(top1_acc, 4)}, Acc@5: {np.round(top5_acc, 4)}")

def eval_conv_teacher(path_to_model, data_type):
    assert data_type in ["CIFAR100", "TinyImageNet", "ImageNetSketch"]

    teacher = ConvTeacherModel(CFG["conv_teacher"])
    teacher.load_state_dict(torch.load(path_to_model))
    teacher.to(DEVICE)
    teacher.eval()

    test_data_c = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_dataset_CIFAR100_model_512x7x7.npy"        , mmap_mode = "r+")
    test_data_t = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_dataset_TinyImageNet_model_1280x7x7.npy"   , mmap_mode = "r+")
    test_data_s = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_dataset_ImageNetSketch_model_2048x7x7.npy" , mmap_mode = "r+")
    test_labels = np.load(f"./embeddings/stage-2/{CFG['dataset']}-7x7/test_{CFG['dataset']}_labels.npy"                                , mmap_mode = "r+")

    criterion  = nn.CrossEntropyLoss()
    testset    = TeacherDataset(test_data_c,  test_data_t,  test_data_s,  test_labels)
    testloader = DataLoader(testset, **CFG["loader"])

    top1_acc, top5_acc = evaluate(teacher, testloader, criterion)
    print(f"[E] Acc@1: {np.round(top1_acc, 4)}, Acc@5: {np.round(top5_acc, 4)}")


if __name__ == "__main__":
    # PATH_TO_CIFAR100_MODEL        = "./weights/students/cifar100/exp-6-identity-2048.pt"
    # PATH_TO_TINY_IMAGENET_MODEL   = "./weights/students/tiny-imagenet/exp-9-identity-2048.pt"
    # PATH_TO_IMAGENET_SKETCH_MODEL   = "./weights/students/imagenet-sketch/exp-6-identity-2048.pt"

    # run(PATH_TO_CIFAR100_MODEL,        data_type = "CIFAR100"       )
    # run(PATH_TO_TINY_IMAGENET_MODEL,   data_type = "TinyImageNet"   )
    # run(PATH_TO_IMAGENET_SKETCH_MODEL,   data_type = "ImageNetSketch" )


    # PATH_TO_CIFAR100_MODEL = "./weights/experts/CIFAR100/exp-16-kd:1.0-t:5-ace-epoch:158-acc@1:0.59.pt"
    # run(PATH_TO_CIFAR100_MODEL, data_type = "CIFAR100")

    # PATH_TO_TINY_IMAGENET_MODEL = "./weights/experts/TinyImageNet/exp-14-kd:0.6-t:2-ace-epoch:194-acc@1:0.51.pt"
    # run(PATH_TO_TINY_IMAGENET_MODEL, data_type = "TinyImageNet")

    # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/ImageNetSketch/exp-10-kd:1.0-t:2-ace-epoch:182-acc@1:0.52.pt"
    # run(PATH_TO_IMAGENET_SKETCH_MODEL, data_type = "ImageNetSketch")

    # PATH_TO_CIFAR100_MODEL = "./weights/students/stage-2/CIFAR100/exp-21-multi-dist-base_epoch_164_acc@1_0.623.pt"
    # run(PATH_TO_CIFAR100_MODEL, data_type = "CIFAR100")

    # PATH_TO_TINY_IMAGENET_MODEL = "./weights/students/stage-2/TinyImageNet/exp-18-multi-dist-base_epoch_199_acc@1_0.517.pt"
    # run(PATH_TO_TINY_IMAGENET_MODEL, data_type = "TinyImageNet")

    # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/students/stage-2/ImageNetSketch/exp-15-multi-dist-base_epoch_196_acc@1_0.613.pt"
    # run(PATH_TO_IMAGENET_SKETCH_MODEL, data_type = "ImageNetSketch")


    # PATH_TO_CIFAR100_MODEL = "./weights/teachers/stage-2/CIFAR100/exp-20-conv-teacher_epoch_283_acc@1_0.633.pt"
    # eval_conv_teacher(PATH_TO_CIFAR100_MODEL, data_type = "CIFAR100")

    # PATH_TO_TINY_IMAGENET_MODEL = "./weights/teachers/stage-2/TinyImageNet/exp-17-conv-teacher_epoch_380_acc@1_0.478.pt"
    # eval_conv_teacher(PATH_TO_TINY_IMAGENET_MODEL, data_type = "TinyImageNet")

    PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/teachers/stage-2/ImageNetSketch/exp-14-conv-teacher_epoch_964_acc@1_0.615.pt"
    eval_conv_teacher(PATH_TO_IMAGENET_SKETCH_MODEL, data_type = "ImageNetSketch")