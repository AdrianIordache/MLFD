from common import *

class TeacherDataset(Dataset):
    def __init__(self, c_data, t_data, s_data, labels):
        super().__init__()
        self.c_data = c_data
        self.t_data = t_data
        self.s_data = s_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ix):
        c_sample = torch.tensor(self.c_data[ix])
        t_sample = torch.tensor(self.t_data[ix])
        s_sample = torch.tensor(self.s_data[ix])
        label    = torch.tensor(self.labels[ix])
        
        inputs   = torch.cat((c_sample, t_sample, s_sample), dim = 0)
        return inputs.float(), label.long()

class CIFAR100Dataset(Dataset):
    def __init__(self, images, labels, transform = None, c_data = None, t_data = None, s_data = None, c_labels = None):
        super().__init__()
        self.images    = images
        self.labels    = labels
        self.transform = transform
        
        self.c_data   = c_data
        self.t_data   = t_data
        self.s_data   = s_data
        self.c_labels = c_labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image_r = image[     : 1024].reshape(32, 32)
        image_g = image[1024 : 2048].reshape(32, 32)
        image_b = image[2048 :     ].reshape(32, 32)

        image = np.dstack((image_r, image_g, image_b))
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
        
        if self.c_data is not None and self.t_data is not None and \
            self.s_data is not None and self.c_labels is not None:

            c_sample = torch.tensor(self.c_data[idx])
            t_sample = torch.tensor(self.t_data[idx])
            s_sample = torch.tensor(self.s_data[idx])
            c_label  = torch.tensor(self.c_labels[idx])

            assert c_label == label, f"{c_label} and {label}"
            maps = torch.cat((c_sample, t_sample, s_sample), dim = 0)
            
            return image.float(), maps.float(), c_label.long()
        else:
            return image.float(), label.long()

class ImageNetDataset(Dataset):
    def __init__(self, data, transform = None, c_data = None, t_data = None, s_data = None, c_labels = None):
        super().__init__()
        self.data      = data
        self.transform = transform

        self.c_data   = c_data
        self.t_data   = t_data
        self.s_data   = s_data
        self.c_labels = c_labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data["path"].values[idx]

        image = cv2.imread(image_path)
        label = self.data["class"].values[idx]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        if self.c_data is not None and self.t_data is not None and \
            self.s_data is not None and self.c_labels is not None:
            
            c_sample = torch.tensor(self.c_data[idx])
            t_sample = torch.tensor(self.t_data[idx])
            s_sample = torch.tensor(self.s_data[idx])
            c_label  = torch.tensor(self.c_labels[idx])

            assert c_label == label, f"{c_label} and {label}"
            maps = torch.cat((c_sample, t_sample, s_sample), dim = 0)
            
            return image.float(), maps.float(), c_label.long()
        else:
            return image.float(), label.long()

def get_dataloaders(config, distillation = False):
    trainset, testset = None, None
    train_c_data, train_t_data, train_s_data, train_c_labels = None, None, None, None

    if distillation:
        train_c_data   = np.load(f"./embeddings/stage-2/{config['dataset']}-7x7/train_{config['dataset']}_dataset_CIFAR100_model_512x7x7.npy"        , mmap_mode = "r+")
        train_t_data   = np.load(f"./embeddings/stage-2/{config['dataset']}-7x7/train_{config['dataset']}_dataset_TinyImageNet_model_1280x7x7.npy"   , mmap_mode = "r+")
        train_s_data   = np.load(f"./embeddings/stage-2/{config['dataset']}-7x7/train_{config['dataset']}_dataset_ImageNetSketch_model_2048x7x7.npy" , mmap_mode = "r+")
        train_c_labels = np.load(f"./embeddings/stage-2/{config['dataset']}-7x7/train_{config['dataset']}_labels.npy"                                , mmap_mode = "r+")

    if config['dataset'] == "CIFAR100":
        means, stds = CIFAR_MEANS, CIFAR_STDS
    else:
        means, stds = IMAGENET_MEANS, IMAGENET_STDS

    if config["use_timm"] == True:
        train_transforms = create_transform(
            input_size   = config["image_size"],
            is_training  = True,
            auto_augment = config["rand_aug"],
            re_prob      = config["re_prob"],
            re_mode      = config["re_mode"],
            re_count     = config["re_count"],
            mean         = means,
            std          = stds,
        )

        test_transforms = create_transform(
            input_size   = config["image_size"],
            is_training  = False,
        )

        # test_transforms = T.Compose([
        #     T.ToPILImage(),
        #     T.Resize((config["image_size"], config["image_size"])),
        #     ] + config['test_transforms'] + [
        #     T.ToTensor(),
        #     T.Normalize(means, stds)
        # ])


        train_transforms.transforms.insert(0, T.ToPILImage())
        test_transforms.transforms.insert(0, T.ToPILImage())

    else:
        train_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((config["image_size"], config["image_size"])),
            ] + config['train_transforms'] + [
            T.ToTensor(),
            T.Normalize(means, stds)
        ])


        test_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((config["image_size"], config["image_size"])),
            ] + config['test_transforms'] + [
            T.ToTensor(),
            T.Normalize(means, stds)
        ])

    print(train_transforms)
    print(test_transforms)

    if config['dataset'] == "CIFAR100":
        train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
        test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)
  
        trainset = CIFAR100Dataset(
            train_images, train_labels, train_transforms, 
            train_c_data, train_t_data, train_s_data, train_c_labels
        )
        testset  = CIFAR100Dataset(test_images, test_labels, test_transforms)

    if config['dataset'] == "TinyImageNet":
        train_tiny_imagenet = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
        test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
        
        trainset = ImageNetDataset(
            train_tiny_imagenet, train_transforms, 
            train_c_data, train_t_data, train_s_data, train_c_labels
        )
        testset  = ImageNetDataset(test_tiny_imagenet, test_transforms)
        
    if config['dataset'] == "ImageNetSketch":
        train_imagenet_sketch = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
        test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
        
        trainset = ImageNetDataset(
            train_imagenet_sketch, train_transforms, 
            train_c_data, train_t_data, train_s_data, train_c_labels
        )
        testset  = ImageNetDataset(test_imagenet_sketch, test_transforms)
        
    trainloader = DataLoader(trainset, **config["trainloader"])
    testloader  = DataLoader(testset, **config["testloader"])
    
    return trainloader, testloader

class TeacherDataset(Dataset):
    def __init__(self, path_to_samples, path_to_labels, nf_c, nf_t):#, nf_s): #, nf_o):
        super().__init__()
        self.path_to_samples = path_to_samples
        self.path_to_labels  = path_to_labels

        self.labels = np.load(self.path_to_labels)
        
        self.nf_c = nf_c
        self.nf_t = nf_t
        # self.nf_s = nf_s
        # self.nf_o = nf_o

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ix):
        path_to_sample = os.path.join(
            self.path_to_samples, 
            f"sample_{self.nf_c}x{self.nf_t}_{ix}.npy"
        )

        sample = np.load(path_to_sample, mmap_mode = "r")
        label  = self.labels[ix]

        sample = torch.tensor(sample)
        label  = torch.tensor(label)
    
        return sample.float(), label.long()

class CIFAR100_Distillation_Dataset(Dataset):
    def __init__(self, images, labels, transform = None, path_to_samples = None, path_to_labels = None, nf_c = None, nf_t = None, nf_s = None):
        super().__init__()
        self.images    = images
        self.labels    = labels
        self.transform = transform

        self.path_to_samples = path_to_samples
        self.path_to_labels  = path_to_labels

        self.nf_c = nf_c
        self.nf_t = nf_t
        self.nf_s = nf_s

        if self.path_to_labels is not None:
            self.emb_labels = np.load(self.path_to_labels)  

        self.distillation = True if self.path_to_samples else False
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image_r = image[     : 1024].reshape(32, 32)
        image_g = image[1024 : 2048].reshape(32, 32)
        image_b = image[2048 :     ].reshape(32, 32)

        image = np.dstack((image_r, image_g, image_b))
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
        
        if self.distillation:
            path_to_sample = os.path.join(
                self.path_to_samples, 
                f"sample_{self.nf_c}x{self.nf_t}x{self.nf_s}_{idx}.npy"
            )

            sample  = np.load(path_to_sample, mmap_mode = "r")
            e_label = self.emb_labels[idx] 

            sample  = torch.tensor(sample)
            e_label = torch.tensor(e_label)

            assert e_label == label, f"{e_label} and {label}"
            
            return image.float(), sample.float(), e_label.long()
        else:
            return image.float(), label.long()

class ImageNet_Distillation_Dataset(Dataset):
    def __init__(self, data, transform = None, path_to_samples = None, path_to_labels = None, nf_c = None, nf_t = None): #, nf_s = None): #, nf_o = None):
        super().__init__()
        self.data      = data
        self.transform = transform

        self.path_to_samples = path_to_samples
        self.path_to_labels  = path_to_labels

        self.nf_c = nf_c
        self.nf_t = nf_t
        # self.nf_s = nf_s
        # self.nf_o = nf_o

        if self.path_to_labels is not None:
            self.emb_labels = np.load(self.path_to_labels)  

        self.distillation = True if self.path_to_samples else False
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data["path"].values[idx]

        image = cv2.imread(image_path)
        label = self.data["class"].values[idx]
        label = torch.tensor(label)

        if self.transform:
            # try:
            image = self.transform(image)
            # except:
            #     print(image)
            #     print(self.data["path"].values[idx])
            #     return

        if self.distillation:
            path_to_sample = os.path.join(
                self.path_to_samples, 
                f"sample_{self.nf_c}x{self.nf_t}_{idx}.npy"
            )

            sample  = np.load(path_to_sample, mmap_mode = "r")
            e_label = self.emb_labels[idx] 

            sample  = torch.tensor(sample)
            e_label = torch.tensor(e_label)

            assert e_label == label, f"{e_label} and {label}"
            
            return image.float(), sample.float(), e_label.long()
        else:
            return image.float(), label.long()


def get_dataloaders_advanced(config, distillation = False):
    trainset, testset  = None, None

    path_to_embeddings    = None
    path_to_train_samples = None
    path_to_train_labels  = None

    if distillation:
        path_to_embeddings = f"./embeddings/stage-6/{config['dataset']}/train/" # f"./embeddings/stage-4/part-3/{config['dataset']}/train/"
        path_to_train_samples = os.path.join(path_to_embeddings, f"{config['e_size']}x{config['e_size']}")
        path_to_train_labels  = os.path.join(path_to_embeddings, f"{config['dataset']}_train_labels.npy")

    if config['dataset'] == "CIFAR100":
        means, stds = CIFAR_MEANS, CIFAR_STDS
    else:
        means, stds = IMAGENET_MEANS, IMAGENET_STDS

    if config["use_timm"] == True:
        train_transforms = create_transform(
            input_size   = config["image_size"],
            is_training  = True,
            auto_augment = config["rand_aug"],
            re_prob      = config["re_prob"],
            re_mode      = config["re_mode"],
            re_count     = config["re_count"],
            mean         = means,
            std          = stds,
        )

        test_transforms = create_transform(
            input_size   = config["image_size"],
            is_training  = False,
        )

        train_transforms.transforms.insert(0, T.ToPILImage())
        test_transforms.transforms.insert(0, T.ToPILImage())

    else:
        train_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((config["image_size"], config["image_size"])),
            ] + config['train_transforms'] + [
            T.ToTensor(),
            T.Normalize(means, stds)
        ])


        test_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((config["image_size"], config["image_size"])),
            ] + config['test_transforms'] + [
            T.ToTensor(),
            T.Normalize(means, stds)
        ])

    if config['dataset'] == "CIFAR100":
        train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
        test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)
  
        trainset = CIFAR100_Distillation_Dataset(
            train_images, train_labels, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )

        testset  = CIFAR100_Distillation_Dataset(test_images, test_labels, test_transforms)

    if config['dataset'] == "TinyImageNet":
        train_tiny_imagenet = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
        test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_tiny_imagenet, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )
        testset  = ImageNet_Distillation_Dataset(test_tiny_imagenet, test_transforms)
        
    if config['dataset'] == "ImageNetSketch":
        train_imagenet_sketch = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)
        test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_imagenet_sketch, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )
        testset  = ImageNet_Distillation_Dataset(test_imagenet_sketch, test_transforms)
        
    if config['dataset'] == "Caltech101":
        train_caltech = pd.read_csv(PATH_TO_CALTECH_TRAIN)
        test_caltech  = pd.read_csv(PATH_TO_CALTECH_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_caltech, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )
        testset  = ImageNet_Distillation_Dataset(test_caltech, test_transforms)
    
    if config['dataset'] == "Flowers102":
        train_flowers = pd.read_csv(PATH_TO_FLOWERS_TRAIN)
        test_flowers  = pd.read_csv(PATH_TO_FLOWERS_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_flowers, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )
        testset  = ImageNet_Distillation_Dataset(test_flowers, test_transforms)
        
    if config['dataset'] == "CUB200":
        train_cub200 = pd.read_csv(PATH_TO_CUB200_TRAIN)
        test_cub200  = pd.read_csv(PATH_TO_CUB200_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_cub200, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )
        testset  = ImageNet_Distillation_Dataset(test_cub200, test_transforms)
        
    if config['dataset'] == "OxfordPets":
        train_pets = pd.read_csv(PATH_TO_PETS_TRAIN)
        test_pets  = pd.read_csv(PATH_TO_PETS_TEST)
        
        trainset = ImageNet_Distillation_Dataset(
            train_pets, train_transforms, 
            path_to_train_samples, path_to_train_labels, **config["n_features_maps"]
        )
        testset  = ImageNet_Distillation_Dataset(test_pets, test_transforms)
        
    trainloader = DataLoader(trainset, **config["trainloader"])
    testloader  = DataLoader(testset, **config["testloader"])
    
    return trainloader, testloader

def get_dataloaders_baseline(config):
    testset = None

    CIFAR100_train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((config["image_size"], config["image_size"])),
        ] + config['train_transforms'] + [
        T.ToTensor(),
        T.Normalize(CIFAR_MEANS, CIFAR_STDS)
    ])

    CIFAR100_test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((config["image_size"], config["image_size"])),
        ] + config['test_transforms'] + [
        T.ToTensor(),
        T.Normalize(CIFAR_MEANS, CIFAR_STDS)
    ])

    ImageNet_train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((config["image_size"], config["image_size"])),
        ] + config['train_transforms'] + [
        T.ToTensor(),
        T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    ImageNet_test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((config["image_size"], config["image_size"])),
        ] + config['test_transforms'] + [
        T.ToTensor(),
        T.Normalize(IMAGENET_MEANS, IMAGENET_STDS)
    ])

    train_images, train_labels = load_cifar(PATH_TO_CIFAR_TRAIN)
    train_tiny_imagenet        = pd.read_csv(PATH_TO_TINY_IMAGENET_TRAIN)
    train_imagenet_sketch      = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TRAIN)

    CIFAR100_trainset       = CIFAR100_Distillation_Dataset(
        train_images, train_labels, CIFAR100_train_transforms, 
    )

    TinyImageNet_trainset   = ImageNet_Distillation_Dataset(
        train_tiny_imagenet, ImageNet_train_transforms, 
    )

    ImageNetSketch_trainset = ImageNet_Distillation_Dataset(
        train_imagenet_sketch, ImageNet_train_transforms, 
    )

    CIFAR100_trainloader       = DataLoader(CIFAR100_trainset, **config["trainloader"])
    TinyImageNet_trainloader   = DataLoader(TinyImageNet_trainset, **config["trainloader"])
    ImageNetSketch_trainloader = DataLoader(ImageNetSketch_trainset, **config["trainloader"])

    if config['dataset'] == "CIFAR100":
        test_images,  test_labels  = load_cifar(PATH_TO_CIFAR_TEST)
        testset  = CIFAR100_Distillation_Dataset(test_images, test_labels, CIFAR100_test_transforms)

    if config['dataset'] == "TinyImageNet":
        test_tiny_imagenet  = pd.read_csv(PATH_TO_TINY_IMAGENET_TEST)
        testset  = ImageNet_Distillation_Dataset(test_tiny_imagenet, ImageNet_test_transforms)
        
    if config['dataset'] == "ImageNetSketch":
        test_imagenet_sketch  = pd.read_csv(PATH_TO_IMAGENET_SKETCH_TEST)
        testset  = ImageNet_Distillation_Dataset(test_imagenet_sketch, ImageNet_test_transforms)
        
    testloader  = DataLoader(testset, **config["testloader"])
    return (CIFAR100_trainloader, TinyImageNet_trainloader, ImageNetSketch_trainloader), testloader