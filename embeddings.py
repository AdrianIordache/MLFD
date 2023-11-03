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
        batch_size     = 768,
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

def get_datasets(data_type, config):
    trainset, testset = None, None

    if data_type == "CIFAR100":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(CIFAR_MEANS, CIFAR_STDS)
        ])

        train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
        test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)

        trainset = CIFAR100Dataset(train_images, train_labels, transforms)
        testset  = CIFAR100Dataset(test_images, test_labels, transforms)

    if data_type == "TinyImageNet":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])

        train_tiny_imagenet = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
        test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
        
        trainset = ImageNetDataset(train_tiny_imagenet, transforms)
        testset  = ImageNetDataset(test_tiny_imagenet, transforms)
        
    if data_type == "ImageNetSketch":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])

        train_imagenet_sketch = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
        test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
    
        trainset = ImageNetDataset(train_imagenet_sketch, transforms)
        testset  = ImageNetDataset(test_imagenet_sketch, transforms)
    
    return trainset, testset

def get_activation(name, out_dict):
    def hook(model, input, output):
        out_dict[name] = output.detach()
    return hook

if __name__ == "__main__":
    PATH_TO_CIFAR100_MODEL        = "./weights/students/cifar100/exp-6-identity-2048.pt"
    PATH_TO_TINY_IMAGENET_MODEL   = "./weights/students/tiny-imagenet/exp-9-identity-2048.pt"
    PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/students/imagenet-sketch/exp-6-identity-2048.pt"

    cifar100_outputs, tiny_imagenet_outputs, imagenet_sketch_outputs = {}, {}, {}

    CIFAR100Model = IntermediateModel(CFG["CIFAR100"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL))
    CIFAR100Model.eval()

    # CIFAR100Model.model.layer2.register_forward_hook(get_activation('size_28',      cifar100_outputs))
    # CIFAR100Model.model.layer3.register_forward_hook(get_activation('size_14',      cifar100_outputs))
    CIFAR100Model.model.layer4.register_forward_hook(get_activation('size_7',       cifar100_outputs))
    # CIFAR100Model.intermediate.register_forward_hook(get_activation('intermediate', cifar100_outputs))

    TinyImageNetModel = IntermediateModel(CFG["TinyImageNet"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
    TinyImageNetModel.eval()

    # TinyImageNetModel.model.blocks[2].register_forward_hook(get_activation('size_28',   tiny_imagenet_outputs))
    # TinyImageNetModel.model.blocks[4].register_forward_hook(get_activation('size_14',   tiny_imagenet_outputs))
    TinyImageNetModel.model.conv_head.register_forward_hook(get_activation('size_7',    tiny_imagenet_outputs))
    # TinyImageNetModel.intermediate.register_forward_hook(get_activation('intermediate', tiny_imagenet_outputs))

    ImageNetSketchModel = IntermediateModel(CFG["ImageNetSketch"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL))
    ImageNetSketchModel.eval()
    
    # ImageNetSketchModel.model.layer2.register_forward_hook(get_activation('size_28',      tiny_imagenet_outputs))
    # ImageNetSketchModel.model.layer3.register_forward_hook(get_activation('size_14',      imagenet_sketch_outputs))
    ImageNetSketchModel.model.layer4.register_forward_hook(get_activation('size_7',       imagenet_sketch_outputs))
    # ImageNetSketchModel.intermediate.register_forward_hook(get_activation('intermediate', imagenet_sketch_outputs))


    for data_type in ["TinyImageNet"]: # "CIFAR100", "TinyImageNet", "ImageNetSketch"
        trainset, testset = get_datasets(data_type, CFG)

        trainloader = DataLoader(trainset, **CFG["loader"])
        testloader  = DataLoader(testset, **CFG["loader"])

        for (mode, loader) in [("train", trainloader), ("test", testloader)]:
            if mode == "train":
                len_dataset = len(trainset)
            else: 
                len_dataset = len(testset)

            # outputs = {
            #     "CIFAR100": {
            #         # "size_28"      : np.zeros((len_dataset, 128, 28, 28)),
            #         # "size_14"      : np.zeros((len_dataset, 256, 14, 14)),
            #         # "size_7"       : np.zeros((len_dataset, 512, 7, 7)),
            #         # "intermediate" : np.zeros((len_dataset, 2048)),
            #     },
            #     "TinyImageNet": {
            #         # "size_28"      : np.zeros((len_dataset, 40, 28, 28)),
            #         # "size_14"      : np.zeros((len_dataset, 112, 14, 14)),
            #         # "size_7"       : np.zeros((len_dataset, 1280, 7, 7)),
            #         # "intermediate" : np.zeros((len_dataset, 2048)),
            #     },
            #     "ImageNetSketch": {
            #         # "size_28"      : np.zeros((len_dataset, 512, 28, 28)),
            #         # "size_14"      : np.zeros((len_dataset, 1024, 14, 14)),
            #         # "size_7"       : np.zeros((len_dataset, 2048, 7, 7)),
            #         # "intermediate" : np.zeros((len_dataset, 2048)),
            #     },
            #     "labels": np.zeros((len_dataset, 1))
            # }

            # cifar100_maps        = np.zeros((len_dataset, 512, 7, 7))
            # tiny_imagenet_maps   = np.zeros((len_dataset, 1280, 7, 7))
            # imagenet_sketch_maps = np.zeros((len_dataset, 2048, 7, 7))
            # labels               = np.zeros((len_dataset,))

            cifar100_maps, tiny_imagenet_maps, imagenet_sketch_maps, labels = [], [], [], []

            start = end = time.time()
            for batch_idx, (images, labels_) in enumerate(loader):
                images  = images.to(DEVICE)
                labels_ = labels_.cpu().numpy()

                # batch_size = images.shape[0]
                with torch.no_grad():
                    _  = CIFAR100Model(images).cpu().detach().numpy()
                    _  = TinyImageNetModel(images).cpu().detach().numpy()
                    _  = ImageNetSketchModel(images).cpu().detach().numpy()

                # outputs["CIFAR100"]["size_28"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]       = cifar100_outputs["size_28"].cpu().detach().numpy()
                # outputs["TinyImageNet"]["size_28"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]   = tiny_imagenet_outputs["size_28"].cpu().detach().numpy()
                # outputs["ImageNetSketch"]["size_28"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :] = tiny_imagenet_outputs["size_28"].cpu().detach().numpy()

                # outputs["CIFAR100"]["size_14"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]       = cifar100_outputs["size_14"].cpu().detach().numpy()
                # outputs["TinyImageNet"]["size_14"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]   = tiny_imagenet_outputs["size_14"].cpu().detach().numpy()
                # outputs["ImageNetSketch"]["size_14"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :] = imagenet_sketch_outputs["size_14"].cpu().detach().numpy()

                # outputs["CIFAR100"]["size_7"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]        = cifar100_outputs["size_7"].cpu().detach().numpy()
                # outputs["TinyImageNet"]["size_7"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]    = tiny_imagenet_outputs["size_7"].cpu().detach().numpy()
                # outputs["ImageNetSketch"]["size_7"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]  = imagenet_sketch_outputs["size_7"].cpu().detach().numpy()

                # outputs["CIFAR100"]["intermediate"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :]        = cifar100_outputs["intermediate"].cpu().detach().numpy()
                # outputs["TinyImageNet"]["intermediate"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :]    = tiny_imagenet_outputs["intermediate"].cpu().detach().numpy()
                # outputs["ImageNetSketch"]["intermediate"][batch_idx * batch_size : (batch_idx + 1) * batch_size, :]  = imagenet_sketch_outputs["intermediate"].cpu().detach().numpy()

                # cifar100_maps[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]        = cifar100_outputs["size_7"].cpu().detach().numpy()
                # tiny_imagenet_maps[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :]   = tiny_imagenet_outputs["size_7"].cpu().detach().numpy()
                # imagenet_sketch_maps[batch_idx * batch_size : (batch_idx + 1) * batch_size, :, :, :] = imagenet_sketch_outputs["size_7"].cpu().detach().numpy()
                # labels[batch_idx * batch_size : (batch_idx + 1) * batch_size] = labels_

                cifar100_maps.extend(cifar100_outputs["size_7"].cpu().detach().numpy())
                tiny_imagenet_maps.extend(tiny_imagenet_outputs["size_7"].cpu().detach().numpy())
                imagenet_sketch_maps.extend(imagenet_sketch_outputs["size_7"].cpu().detach().numpy())
                labels.extend(labels_)

                end = time.time()
                if (batch_idx + 1) % CFG['print_freq'] == 0 or (batch_idx + 1) == len(loader):
                    message = f"[G] B: [{batch_idx + 1}/{len(loader)}], " + \
                              f"{time_since(start, float(batch_idx + 1) / len(loader))}"

                    print(message)

            cifar100_maps        = np.array(cifar100_maps)
            tiny_imagenet_maps   = np.array(tiny_imagenet_maps) 
            imagenet_sketch_maps = np.array(imagenet_sketch_maps)
            labels               = np.array(labels)

            with open(f"./embeddings/stage-2/{data_type}-7x7/{mode}_{data_type}_dataset_CIFAR100_model_512x7x7.npy", 'wb') as fp:
                  np.save(fp, cifar100_maps)

            with open(f"./embeddings/stage-2/{data_type}-7x7/{mode}_{data_type}_dataset_TinyImageNet_model_1280x7x7.npy", 'wb') as fp:
                  np.save(fp, tiny_imagenet_maps)

            with open(f"./embeddings/stage-2/{data_type}-7x7/{mode}_{data_type}_dataset_ImageNetSketch_model_2048x7x7.npy", 'wb') as fp:
                  np.save(fp, imagenet_sketch_maps)

            with open(f"./embeddings/stage-2/{data_type}-7x7/{mode}_{data_type}_labels.npy", 'wb') as fp:
                  np.save(fp, labels)




