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
        num_workers    = 4,
        pin_memory     = True,
        sampler        = sampler, 
        worker_init_fn = seed_worker, 
        drop_last      = False
    ),

    n_embedding = 2048,
    activation  = nn.Identity(),
    image_size = 224,
    print_freq = 10
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

if __name__ == "__main__":
    # PATH_TO_CIFAR100_MODEL        = "./weights/students/cifar100/exp-6-identity-2048.pt"
    # PATH_TO_TINY_IMAGENET_MODEL   = "./weights/students/tiny-imagenet/exp-9-identity-2048.pt"
    # PATH_TO_IMAGENET_SKETCH_MODEL   = "./weights/students/imagenet-sketch/exp-6-identity-2048.pt"

    # run(PATH_TO_CIFAR100_MODEL,        data_type = "CIFAR100"       )
    # run(PATH_TO_TINY_IMAGENET_MODEL,   data_type = "TinyImageNet"   )
    # run(PATH_TO_IMAGENET_SKETCH_MODEL,   data_type = "ImageNetSketch" )


    PATH_TO_CIFAR100_MODEL = "./weights/experts/CIFAR100/exp-16-kd:1.0-t:5-ace-epoch:158-acc@1:0.59.pt"
    run(PATH_TO_CIFAR100_MODEL, data_type = "CIFAR100")

    PATH_TO_TINY_IMAGENET_MODEL = "./weights/experts/TinyImageNet/exp-14-kd:0.6-t:2-ace-epoch:194-acc@1:0.51.pt"
    run(PATH_TO_TINY_IMAGENET_MODEL, data_type = "TinyImageNet")

    PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/ImageNetSketch/exp-10-kd:1.0-t:2-ace-epoch:182-acc@1:0.52.pt"
    run(PATH_TO_IMAGENET_SKETCH_MODEL, data_type = "ImageNetSketch")