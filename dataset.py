from common import *

class CIFAR100Dataset(Dataset):
    def __init__(self, images: np.array, labels: np.array, transform = None):
        super().__init__()
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image_r = image[     : 1024].reshape(32, 32)
        image_g = image[1024 : 2048].reshape(32, 32)
        image_b = image[2048 :     ].reshape(32, 32)

        image = np.dstack((image_r, image_g, image_b))

        if self.transform:
            image = self.transform(image)
        
        return image, label     

class ImageNetDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform = None):
        super().__init__()
        self.data      = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data["path"].values[idx]

        image = cv2.imread(image_path)
        label = self.data["class"].values[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label 
