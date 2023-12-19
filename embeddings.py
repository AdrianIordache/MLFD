from common        import *
from models        import *
from dataset       import *
from labels        import *
from procedures    import evaluate

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
        batch_size     = 1,
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
    PATH_TO_CIFAR100_MODEL        = "./weights/experts/stage-1/CIFAR100/exp-6-identity-2048.pt"
    PATH_TO_TINY_IMAGENET_MODEL   = "./weights/experts/stage-1/TinyImageNet/exp-9-identity-2048.pt"
    PATH_TO_IMAGENET_SKETCH_MODEL = "./weights/experts/stage-1/ImageNetSketch/exp-6-identity-2048.pt"

    cifar100_outputs, tiny_imagenet_outputs, imagenet_sketch_outputs = {}, {}, {}

    CIFAR100Model = IntermediateModel(CFG["CIFAR100"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    CIFAR100Model.load_state_dict(torch.load(PATH_TO_CIFAR100_MODEL))
    CIFAR100Model.eval()

    CIFAR100Model.model.layer2.register_forward_hook(get_activation('size_28', cifar100_outputs))
    CIFAR100Model.model.layer3.register_forward_hook(get_activation('size_14', cifar100_outputs))
    CIFAR100Model.model.layer4.register_forward_hook(get_activation('size_7',  cifar100_outputs))
    CIFAR100Model.intermediate.register_forward_hook(get_activation('size_1',  cifar100_outputs))

    TinyImageNetModel = IntermediateModel(CFG["TinyImageNet"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    TinyImageNetModel.load_state_dict(torch.load(PATH_TO_TINY_IMAGENET_MODEL))
    TinyImageNetModel.eval()

    TinyImageNetModel.model.blocks[2].register_forward_hook(get_activation('size_28', tiny_imagenet_outputs))
    TinyImageNetModel.model.blocks[4].register_forward_hook(get_activation('size_14', tiny_imagenet_outputs))
    TinyImageNetModel.model.conv_head.register_forward_hook(get_activation('size_7',  tiny_imagenet_outputs))
    TinyImageNetModel.intermediate.register_forward_hook(get_activation('size_1',     tiny_imagenet_outputs))

    ImageNetSketchModel = IntermediateModel(CFG["ImageNetSketch"], CFG["n_embedding"], CFG["activation"]).to(DEVICE)
    ImageNetSketchModel.load_state_dict(torch.load(PATH_TO_IMAGENET_SKETCH_MODEL))
    ImageNetSketchModel.eval()
    
    ImageNetSketchModel.model.layer2.register_forward_hook(get_activation('size_28', imagenet_sketch_outputs))
    ImageNetSketchModel.model.layer3.register_forward_hook(get_activation('size_14', imagenet_sketch_outputs))
    ImageNetSketchModel.model.layer4.register_forward_hook(get_activation('size_7',  imagenet_sketch_outputs))
    ImageNetSketchModel.intermediate.register_forward_hook(get_activation('size_1',  imagenet_sketch_outputs))


    for data_type in ["CIFAR100", "TinyImageNet", "ImageNetSketch"]: # "CIFAR100", "TinyImageNet", "ImageNetSketch"
        trainset, testset = get_datasets(data_type, CFG)

        trainloader = DataLoader(trainset, **CFG["loader"])
        testloader  = DataLoader(testset, **CFG["loader"])

        for (mode, loader) in [("train", trainloader), ("test", testloader)]:
            counter = 0
            start = end = time.time()

            PATH_TO_EMBEDDINGS = f"./embeddings/stage-3/{data_type}/{mode}"
            os.makedirs(PATH_TO_EMBEDDINGS, exist_ok = True)

            PATHS = {}
            for size in [1, 7, 14, 28]:
                PATHS[f"size_{size}"] = f"{PATH_TO_EMBEDDINGS}/{size}x{size}/"
                os.makedirs(PATHS[f"size_{size}"], exist_ok = True)

            labels = []
            for batch_idx, (image, label) in enumerate(loader):
                image = image.to(DEVICE)
    
                label = label.item()
                labels.append(label)

                # with torch.no_grad():
                #     _  = CIFAR100Model(image).cpu().detach().numpy()
                #     _  = TinyImageNetModel(image).cpu().detach().numpy()
                #     _  = ImageNetSketchModel(image).cpu().detach().numpy()

                # embeddings = {
                #     "image" : image.cpu().detach().numpy(),
                #     "label" : label.cpu().detach().numpy(),

                #     "CIFAR100" : {
                #         "size_28" : cifar100_outputs["size_28"].cpu().detach().numpy(),
                #         "size_14" : cifar100_outputs["size_14"].cpu().detach().numpy(),
                #         "size_7"  : cifar100_outputs["size_7"].cpu().detach().numpy(),
                #         "size_1"  : cifar100_outputs["size_1"].cpu().detach().numpy().squeeze(0),
                #     },

                #     "TinyImageNet" : {
                #         "size_28" : tiny_imagenet_outputs["size_28"].cpu().detach().numpy(),
                #         "size_14" : tiny_imagenet_outputs["size_14"].cpu().detach().numpy(),
                #         "size_7"  : tiny_imagenet_outputs["size_7"].cpu().detach().numpy(),
                #         "size_1"  : tiny_imagenet_outputs["size_1"].cpu().detach().numpy(),
                #     },

                #     "ImageNetSketch" : {
                #         "size_28" : imagenet_sketch_outputs["size_28"].cpu().detach().numpy(),
                #         "size_14" : imagenet_sketch_outputs["size_14"].cpu().detach().numpy(),
                #         "size_7"  : imagenet_sketch_outputs["size_7"].cpu().detach().numpy(),
                #         "size_1"  : imagenet_sketch_outputs["size_1"].cpu().detach().numpy(),
                #     }
                # }

                # sample = {}
                # for size in [1, 7, 14, 28]:
                #     c_size = cifar100_outputs[f"size_{size}"].cpu().detach().squeeze(0).shape[0]
                #     t_size = tiny_imagenet_outputs[f"size_{size}"].cpu().detach().squeeze(0).shape[0]
                #     s_size = imagenet_sketch_outputs[f"size_{size}"].cpu().detach().squeeze(0).shape[0]

                #     sample[f"size_{size}"] = torch.cat((
                #         cifar100_outputs[f"size_{size}"].cpu().detach().squeeze(0),
                #         tiny_imagenet_outputs[f"size_{size}"].cpu().detach().squeeze(0),
                #         imagenet_sketch_outputs[f"size_{size}"].cpu().detach().squeeze(0),
                #     ), dim = 0).numpy()

                #     with open(f"{PATHS[f'size_{size}']}/sample_{c_size}x{t_size}x{s_size}_{counter}.npy", "wb") as handler:
                #         np.save(handler, sample[f"size_{size}"])

                # counter += 1

                # end = time.time()
                # if (batch_idx + 1) % CFG['print_freq'] == 0 or (batch_idx + 1) == len(loader):
                #     message = f"[G] B: [{batch_idx + 1}/{len(loader)}], " + \
                #               f"{time_since(start, float(batch_idx + 1) / len(loader))}"

                #     print(message)

            labels = np.array(labels)
            with open(f"{PATH_TO_EMBEDDINGS}/{data_type}_{mode}_labels.npy", "wb") as handler:
                np.save(handler, labels)

# def get_tensor_shape(dataloader, models, outputs, model_key, size):
#     image, _ = next(iter(dataloader))
#     image    = image.to(DEVICE)
#     _        = models[model_key](image)
#     tensor   = outputs[model_key][f"size_{size}"]
#     return tensor.shape
# 
#     MODELS = {
#         "CIFAR100"       : CIFAR100Model,
#         "TinyImageNet"   : TinyImageNetModel,
#         "ImageNetSketch" : ImageNetSketchModel
#     }

#     OUTPUTS = {
#         "CIFAR100"       : cifar100_outputs,
#         "TinyImageNet"   : tiny_imagenet_outputs,
#         "ImageNetSketch" : imagenet_sketch_outputs
#     }

#     BS = CFG["loader"]["batch_size"]
#         for (mode, loader) in [("TRAIN", trainloader)]: # , ("TEST", testloader)]:
#             if mode == "TRAIN":
#                 len_dataset = len(trainset)
#             else: 
#                 len_dataset = len(testset)

#             labels = np.zeros((len_dataset, ), dtype = np.float16)
#             for batch_idx, (_, labels_) in enumerate(loader):
#                 labels_ = labels_.cpu().numpy()
#                 curr_bs = labels_.shape[0]
#                 step_bs = batch_idx * BS
                
#                 labels[step_bs : step_bs + curr_bs] = labels_

#             PATH_TO_EMBEDDINGS_BASE = f"./embeddings/stage-2/{data_type}"
#             os.makedirs(PATH_TO_EMBEDDINGS_BASE, exist_ok = True)

#             labels = np.array(labels)
#             with open(f"{PATH_TO_EMBEDDINGS_BASE}/{mode}_{data_type}_labels.npy", 'wb') as fp:
#                 np.save(fp, labels)

#             del labels
#             gc.collect()

#             for size in [14]: # [1, 7, 14]
#                 for model_key in ["ImageNetSketch"]:

#                     _, c, w, h = get_tensor_shape(copy.deepcopy(loader), MODELS, OUTPUTS, model_key, size)
                    
#                     maps = np.zeros((len_dataset, c, w, h), dtype = np.float16)
#                     print(maps.shape)

#                     start = end = time.time()
#                     for batch_idx, (images, _) in enumerate(loader):
#                         images  = images.to(DEVICE)

#                         curr_bs = images.shape[0]
#                         step_bs = batch_idx * BS

#                         with torch.no_grad():
#                             _  = MODELS[model_key](images).cpu().detach().numpy()

#                         maps[step_bs : step_bs + curr_bs, :, :, :] = OUTPUTS[model_key][f"size_{size}"].cpu().detach().numpy()
                     
#                         end = time.time()
#                         if (batch_idx + 1) % CFG['print_freq'] == 0 or (batch_idx + 1) == len(loader):
#                             message = f"[G] D/T/M/S/B: [{data_type}][{mode}][S-{size}][{model_key}][{batch_idx + 1}/{len(loader)}], " + \
#                                       f"{time_since(start, float(batch_idx + 1) / len(loader))}"

#                             print(message)

#                     PATH_TO_EMBEDDINGS_MAPS = f"{PATH_TO_EMBEDDINGS_BASE}/{size}x{size}"
#                     os.makedirs(PATH_TO_EMBEDDINGS_MAPS, exist_ok = True)

#                     maps = np.array(maps)
#                     with open(f"{PATH_TO_EMBEDDINGS_MAPS}/{mode}_dataset_{data_type}_model_{model_key}_{c}x{size}x{size}.npy", 'wb') as fp:
#                           np.save(fp, maps)

#                     del maps
#                     free_gpu_memory(DEVICE)
#                     gc.collect()








