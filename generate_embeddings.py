from common        import *
from models        import *
from dataset       import *
from labels        import *
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
 
if __name__ == "__main__":
    PATH_TO_CIFAR100_MODEL        = "./weights/students/cifar100/exp-6-identity-2048.pt"
    PATH_TO_TINY_IMAGENET_MODEL   = "./weights/students/tiny-imagenet/exp-9-identity-2048.pt"
    PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/students/imagenet-sketch/exp-6-identity-2048.pt"

    CIFAR100Model = IntermediateModel(CFG["CIFAR100"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL))
    CIFAR100Model.eval()

    TinyImageNetModel = IntermediateModel(CFG["TinyImageNet"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
    TinyImageNetModel.eval()

    ImageNetSketchModel = IntermediateModel(CFG["ImageNetSketch"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL))
    ImageNetSketchModel.eval()
    
    criterion  = nn.CrossEntropyLoss()
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((CFG["image_size"], CFG["image_size"])),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    trainset, testset = None, None
    
    for data_type in ["cifar100", "tiny_imagenet", "imagenet_sketch"]:
        if data_type == "cifar100":
            train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
            test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)

            trainset = CIFAR100Dataset(train_images, train_labels, transforms)
            testset  = CIFAR100Dataset(test_images, test_labels, transforms)

        if data_type == "tiny_imagenet":
            train_tiny_imagenet = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
            test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
            
            trainset = ImageNetDataset(train_tiny_imagenet, transforms)
            testset  = ImageNetDataset(test_tiny_imagenet, transforms)
            
        if data_type == "imagenet_sketch":
            train_imagenet_sketch = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
            test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
            
            trainset = ImageNetDataset(train_imagenet_sketch, transforms)
            testset  = ImageNetDataset(test_imagenet_sketch, transforms)
            
        trainloader = DataLoader(trainset, **CFG["loader"])
        testloader  = DataLoader(testset, **CFG["loader"])

        for (mode, loader) in [("train", trainloader), ("test", testloader)]:
            embeddings = []
            labels     = []

            start  = end = time.time()
            for batch, (images, labels_) in enumerate(loader):
                images  = images.to(DEVICE)
                labels_ = labels_.cpu().numpy()

                with torch.no_grad():
                    cifar100_outputs        = CIFAR100Model(images, embeddings = True).cpu().numpy()
                    tiny_imagenet_outputs   = TinyImageNetModel(images, embeddings = True).cpu().numpy()
                    imagenet_sketch_outputs = ImageNetSketchModel(images, embeddings = True).cpu().numpy()

                outputs = np.concatenate((cifar100_outputs, tiny_imagenet_outputs, imagenet_sketch_outputs), axis = 1)
                
                embeddings.extend(outputs)
                labels.extend(labels_)
                
                end = time.time()
                if (batch + 1) % CFG['print_freq'] == 0 or (batch + 1) == len(loader):
                    message = f"[G] B: [{batch + 1}/{len(loader)}], " + \
                              f"{time_since(start, float(batch + 1) / len(loader))}"

                    print(message)

            embeddings = np.array(embeddings)
            labels     = np.array(labels)

            columns = [f"cifar_x{i}" for i in range(0, 2048)] + \
                      [f"tiny_imagenet_x{i}" for i in range(0, 2048)] + \
                      [f"imagenet_sketch_x{i}" for i in range(0, 2048)]

            data = pd.DataFrame(embeddings, columns = columns)
            data["label"] = labels
            display(data)

            data.to_csv(f"./embeddings/{data_type}_{mode}_embeddings.csv", index = False)



    



