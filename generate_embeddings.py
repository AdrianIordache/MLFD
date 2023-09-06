from common      import *
from models      import *
from dataset     import *
from labels      import *
from procedures  import evaluate

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
        batch_size     = 24,
        shuffle        = False, 
        num_workers    = 1,
        pin_memory     = True,
        sampler        = sampler, 
        worker_init_fn = seed_worker, 
        drop_last      = False
    ),

    image_size = 224,
)
    

if __name__ == "__main__":
    PATH_TO_CIFAR100_MODEL        = "./weights/cifar100/exp-4-pretrained.pt"
    PATH_TO_TINY_IMAGENET_MODEL   = "./weights/tiny-imagenet/exp-8-efficientnet.pt"
    PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/imagenet-sketch/exp-4-pretrained.pt"

    CIFAR100Model = BaselineModel(CFG["CIFAR100"]).to(DEVICE)
    CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL))
    CIFAR100Model.eval()

    # TinyImageNetModel = BaselineModel(CFG["TinyImageNet"]).to(DEVICE)
    # TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
    # TinyImageNetModel.eval()

    # ImageNetSketchModel = BaselineModel(CFG["ImageNetSketch"]).to(DEVICE)
    # ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL))
    # ImageNetSketchModel.eval()

    # train_df = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
    test_df  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
    
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((CFG["image_size"], CFG["image_size"])),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    testset    = ImageNetDataset(test_df, transforms)
    testloader = DataLoader(testset, **CFG["loader"])

    criterion = nn.CrossEntropyLoss()

    print(evaluate(CIFAR100Model, testloader, criterion))



