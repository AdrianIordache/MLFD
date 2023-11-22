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