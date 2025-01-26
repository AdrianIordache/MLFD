from common        import *
from models        import *
from dataset       import *
from labels        import *
from procedures    import evaluate

CFG = dict(
    advanced = False,

    CIFAR100_Advanced = dict(
        model_name     = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        num_classes    = 100,
        pretrained     = False,
        drop_rate      = 0.0,
        drop_path_rate = 0.6, # stochastic depth
    ),    

    TinyImageNet_Advanced = dict(
        model_name     = "swinv2_tiny_window8_256.ms_in1k",
        num_classes    = 200,
        pretrained     = False,
        drop_rate      = 0.4,
        drop_path_rate = 0.5, # stochastic depth
    ),    

    ImageNetSketch_Advanced = dict(
        model_name     = "fastvit_sa24.apple_in1k",
        num_classes    = 1000,
        pretrained     = False,
        drop_rate      = 0.2,
        drop_path_rate = 0.3, # stochastic depth
    ),    

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

    Caltech101 = dict(
        model_name  = "resnet18",
        num_classes = 102,
        pretrained  = False,
    ),

    Flowers102 = dict(
        model_name  = "tf_efficientnet_b0.in1k",
        num_classes = 102,
        pretrained  = False,
    ),

    CUB200 = dict(
        model_name  = "seresnext26t_32x4d",
        num_classes = 200,
        pretrained  = False,
    ),

    OxfordPets = dict(
        model_name  = "resnet18",
        num_classes = 37,
        pretrained  = False,
    ),

    loader = dict(
        batch_size     = 1,
        shuffle        = False, 
        num_workers    = 4,
        pin_memory     = True,
        sampler        = sampler, 
        worker_init_fn = seed_worker, 
        drop_last      = False
    ),


    n_embedding = 2048, # 2048, 768 
    activation  = nn.Identity(),
    image_size = 224, # 224, 256
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
    
    if data_type == "Caltech101":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])

        train_caltech = pd.read_csv(PATH_TO_CALTECH_TRAIN)
        test_caltech  = pd.read_csv(PATH_TO_CALTECH_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_caltech, transforms, 
        )
        testset  = ImageNet_Distillation_Dataset(test_caltech, transforms)
    
    if data_type == "Flowers102":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])

        train_flowers = pd.read_csv(PATH_TO_FLOWERS_TRAIN)
        test_flowers  = pd.read_csv(PATH_TO_FLOWERS_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_flowers, transforms, 
        )
        testset  = ImageNet_Distillation_Dataset(test_flowers, transforms)
        
    if data_type == "CUB200":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])

        train_cub = pd.read_csv(PATH_TO_CUB200_TRAIN)
        test_cub  = pd.read_csv(PATH_TO_CUB200_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_cub, transforms, 
        )
        testset  = ImageNet_Distillation_Dataset(test_cub, transforms)
        
    if data_type == "OxfordPets":
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((CFG["image_size"], CFG["image_size"])),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
        ])

        train_pets = pd.read_csv(PATH_TO_PETS_TRAIN)
        test_pets  = pd.read_csv(PATH_TO_PETS_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_pets, transforms, 
        )
        testset  = ImageNet_Distillation_Dataset(test_pets, transforms)
        
    return trainset, testset

def get_activation(name, out_dict):
    def hook(model, input, output):
        out_dict[name] = output.detach()
    return hook

if __name__ == "__main__":
    cifar100_outputs, tiny_imagenet_outputs, imagenet_sketch_outputs, other_outputs = {}, {}, {}, {}

    if CFG["advanced"] == True:
        # PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-3/CIFAR100/exp-12-lr-1e-3-sth-0.6_epoch_286_acc@1_0.787.pt"
        # PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-3/TinyImageNet/exp-9-200-epochs_epoch_196_acc@1_0.669.pt"
        # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-3/ImageNetSketch/exp-12-drop-0.2-e200_epoch_192_acc@1_0.724.pt"

        # PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-3/CIFAR100/exp-14-pretrained-small-lr_epoch_148_acc@1_0.914.pt"
        # PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-3/TinyImageNet/exp-10-pretrained_epoch_171_acc@1_0.863.pt"
        # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-3/ImageNetSketch/exp-11-pretrained_epoch_176_acc@1_0.829.pt"

        PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-3/CIFAR100/exp-15-expert-1e-5_epoch_295_acc@1_0.657.pt"
        PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-3/TinyImageNet/exp-27-one-cycle-0.1_epoch_147_acc@1_0.635.pt"
        PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-3/ImageNetSketch/exp-13-expert-1e-4_epoch_188_acc@1_0.677.pt"

        CIFAR100Model = ExpertModel(CFG["CIFAR100_Advanced"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL, map_location = torch.device('cpu')))
        CIFAR100Model.eval()

        CIFAR100Model.model.stages[2].register_forward_hook(get_activation('size_16', cifar100_outputs))
        CIFAR100Model.model.stages[3].register_forward_hook(get_activation('size_8',  cifar100_outputs))
        CIFAR100Model.model.head.register_forward_hook(get_activation('size_1',       cifar100_outputs))

        TinyImageNetModel = ExpertModel(CFG["TinyImageNet_Advanced"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL, map_location = torch.device('cpu')))
        TinyImageNetModel.eval()

        TinyImageNetModel.model.layers[2].register_forward_hook(get_activation('size_16', tiny_imagenet_outputs))
        TinyImageNetModel.model.layers[3].register_forward_hook(get_activation('size_8',  tiny_imagenet_outputs))
        TinyImageNetModel.model.head.register_forward_hook(get_activation('size_1',       tiny_imagenet_outputs))

        ImageNetSketchModel = ExpertModel(CFG["ImageNetSketch_Advanced"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL, map_location = torch.device('cpu')))
        ImageNetSketchModel.eval()

        ImageNetSketchModel.model.stages[2].register_forward_hook(get_activation('size_16', imagenet_sketch_outputs))
        ImageNetSketchModel.model.stages[3].register_forward_hook(get_activation('size_8',  imagenet_sketch_outputs))
        ImageNetSketchModel.latent_proj.register_forward_hook(get_activation('size_1',       imagenet_sketch_outputs))

    else:
        # PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-1/CIFAR100/exp-6-identity-2048.pt"
        # PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-1/TinyImageNet/exp-9-identity-2048.pt"
        # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-1/ImageNetSketch/exp-6-identity-2048.pt"

        # CIFAR100Model = IntermediateModel(CFG["CIFAR100"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        # CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL))
        # CIFAR100Model.eval()

        # CIFAR100Model.model.layer2.register_forward_hook(get_activation('size_28', cifar100_outputs))
        # CIFAR100Model.model.layer3.register_forward_hook(get_activation('size_14', cifar100_outputs))
        # CIFAR100Model.model.layer4.register_forward_hook(get_activation('size_7',  cifar100_outputs))
        # CIFAR100Model.intermediate.register_forward_hook(get_activation('size_1',  cifar100_outputs))

        # TinyImageNetModel = IntermediateModel(CFG["TinyImageNet"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        # TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
        # TinyImageNetModel.eval()

        # TinyImageNetModel.model.blocks[2].register_forward_hook(get_activation('size_28', tiny_imagenet_outputs))
        # TinyImageNetModel.model.blocks[4].register_forward_hook(get_activation('size_14', tiny_imagenet_outputs))
        # TinyImageNetModel.model.conv_head.register_forward_hook(get_activation('size_7',  tiny_imagenet_outputs))
        # TinyImageNetModel.intermediate.register_forward_hook(get_activation('size_1',     tiny_imagenet_outputs))

        # ImageNetSketchModel = IntermediateModel(CFG["ImageNetSketch"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        # ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL))
        # ImageNetSketchModel.eval()
        
        # ImageNetSketchModel.model.layer2.register_forward_hook(get_activation('size_28', imagenet_sketch_outputs))
        # ImageNetSketchModel.model.layer3.register_forward_hook(get_activation('size_14', imagenet_sketch_outputs))
        # ImageNetSketchModel.model.layer4.register_forward_hook(get_activation('size_7',  imagenet_sketch_outputs))
        # ImageNetSketchModel.intermediate.register_forward_hook(get_activation('size_1',  imagenet_sketch_outputs))


        # PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-4/Caltech101/exp-1-baseline_epoch_286_acc@1_0.748.pt"
        # PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-4/Flowers102/exp-1-baseline_epoch_188_acc@1_0.78.pt"
        # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-4/CUB200/exp-1-baseline_epoch_288_acc@1_0.514.pt"
        # PATH_TO_OTHER_MODEL           = "./weights/experts/stage-4/OxfordPets/exp-1-baseline_epoch_274_acc@1_0.639.pt"



        PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-4/Caltech101/exp-1-baseline_epoch_286_acc@1_0.748.pt" # "./weights/experts/stage-5/ImageNetSketch/exp-29-ensemble-resnet_epoch_159_acc@1_0.227.pt"
        PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-4/OxfordPets/exp-1-baseline_epoch_274_acc@1_0.639.pt" # "./weights/experts/stage-5/ImageNetSketch/exp-30-ensemble-efficientnet_epoch_193_acc@1_0.292.pt"
        # PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-4/CUB200/exp-1-baseline_epoch_288_acc@1_0.514.pt" # "./weights/experts/stage-1/ImageNetSketch/exp-6-identity-2048.pt"
        # PATH_TO_OTHER_MODEL           = "./weights/experts/stage-4/OxfordPets/exp-1-baseline_epoch_274_acc@1_0.639.pt"

        # CFG["ImageNetSketch"]["model_name"] = "resnet18"
        CIFAR100Model = ExpertModel(CFG["Caltech101"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL))
        CIFAR100Model.eval()

        # CIFAR100Model.model.layer2.register_forward_hook(get_activation('size_28', cifar100_outputs))
        # CIFAR100Model.model.layer3.register_forward_hook(get_activation('size_14', cifar100_outputs))
        CIFAR100Model.model.layer4.register_forward_hook(get_activation('size_7',  cifar100_outputs))
        # CIFAR100Model.intermediate.register_forward_hook(get_activation('size_1',  cifar100_outputs))


        TinyImageNetModel = ExpertModel(CFG["OxfordPets"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
        TinyImageNetModel.eval()

        TinyImageNetModel.model.layer4.register_forward_hook(get_activation('size_7',  tiny_imagenet_outputs))

        # CFG["ImageNetSketch"]["model_name"] = "tf_efficientnet_b0"
        # TinyImageNetModel = ExpertModel(CFG["Flowers102"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        # TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
        # TinyImageNetModel.eval()

        # TinyImageNetModel.model.blocks[2].register_forward_hook(get_activation('size_28', tiny_imagenet_outputs))
        # TinyImageNetModel.model.blocks[4].register_forward_hook(get_activation('size_14', tiny_imagenet_outputs))
        # TinyImageNetModel.model.conv_head.register_forward_hook(get_activation('size_7',  tiny_imagenet_outputs))
        # TinyImageNetModel.intermediate.register_forward_hook(get_activation('size_1',     tiny_imagenet_outputs))

        # CFG["ImageNetSketch"]["model_name"] = "seresnext26t_32x4d"
        # ImageNetSketchModel = ExpertModel(CFG["CUB200"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        # ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL))
        # ImageNetSketchModel.eval()

        # ImageNetSketchModel.model.layer2.register_forward_hook(get_activation('size_28', imagenet_sketch_outputs))
        # ImageNetSketchModel.model.layer3.register_forward_hook(get_activation('size_14', imagenet_sketch_outputs))
        # ImageNetSketchModel.model.layer4.register_forward_hook(get_activation('size_7',  imagenet_sketch_outputs))
        # ImageNetSketchModel.intermediate.register_forward_hook(get_activation('size_1',  imagenet_sketch_outputs))

        # OtherModel = ExpertModel(CFG["OxfordPets"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
        # OtherModel.load_state_dict(torch.load(PATH_TO_OTHER_MODEL))
        # OtherModel.eval()

        # OtherModel.model.layer4.register_forward_hook(get_activation('size_7',  other_outputs))

    SIZES = []
    if CFG["advanced"] == True:
        SIZES = [1, 8, 16]
    else:
        SIZES = [7]

    for data_type in ["Caltech101", "OxfordPets"]: # "Caltech101", "Flowers102", "CUB200", "OxfordPets" # "CIFAR100", "TinyImageNet", "ImageNetSketch"
        trainset, testset = get_datasets(data_type, CFG)

        trainloader = DataLoader(trainset, **CFG["loader"])
        testloader  = DataLoader(testset, **CFG["loader"])

        for (mode, loader) in [("train", trainloader), ("test", testloader)]:
            counter = 0
            start = end = time.time()

            PATH_TO_EMBEDDINGS = f"./embeddings/stage-6/{data_type}/{mode}" # f"./embeddings/stage-4/part-3/{data_type}/{mode}"
            os.makedirs(PATH_TO_EMBEDDINGS, exist_ok = True)

            PATHS = {}
            for size in SIZES:
                PATHS[f"size_{size}"] = f"{PATH_TO_EMBEDDINGS}/{size}x{size}/"
                os.makedirs(PATHS[f"size_{size}"], exist_ok = True)

            labels = []
            for batch_idx, (image, label) in enumerate(loader):
                image = image.to(DEVICE)
    
                label = label.item()
                labels.append(label)

                with torch.no_grad():
                    _  = CIFAR100Model(image).cpu().detach().numpy()
                    _  = TinyImageNetModel(image).cpu().detach().numpy()
                    # _  = ImageNetSketchModel(image).cpu().detach().numpy()
                    # _  = OtherModel(image).cpu().detach().numpy()

                sample = {}
                for size in SIZES:
                    cifar100_sample = cifar100_outputs[f"size_{size}"].cpu().detach().squeeze(0)
                    tiny_imagenet_sample = tiny_imagenet_outputs[f"size_{size}"].cpu().detach().squeeze(0)
                    # imagenet_sketch_sample = imagenet_sketch_outputs[f"size_{size}"].cpu().detach().squeeze(0)
                    # other_sample = other_outputs[f"size_{size}"].cpu().detach().squeeze(0)

                    if CFG["advanced"] == True and len(tiny_imagenet_sample.shape) != 1:
                        tiny_imagenet_sample = torch.permute(tiny_imagenet_sample, (2, 0, 1))
                    
                    c_size = cifar100_sample.shape[0]
                    t_size = tiny_imagenet_sample.shape[0]
                    # s_size = imagenet_sketch_sample.shape[0]
                    # o_size = other_sample.shape[0]

                    sample[f"size_{size}"] = torch.cat((
                        cifar100_sample, tiny_imagenet_sample #, imagenet_sketch_sample #, other_sample
                    ), dim = 0).numpy()

                    with open(f"{PATHS[f'size_{size}']}/sample_{c_size}x{t_size}_{counter}.npy", "wb") as handler:
                        np.save(handler, sample[f"size_{size}"])

                counter += 1

                end = time.time()
                if (batch_idx + 1) % CFG['print_freq'] == 0 or (batch_idx + 1) == len(loader):
                    message = f"[G] B: [{batch_idx + 1}/{len(loader)}], " + \
                              f"{time_since(start, float(batch_idx + 1) / len(loader))}"

                    print(message)

            labels = np.array(labels)
            with open(f"{PATH_TO_EMBEDDINGS}/{data_type}_{mode}_labels.npy", "wb") as handler:
                np.save(handler, labels)






