from common import *

class CIFAR100Dataset(Dataset):
    def __init__(self, images: np.array, labels: np.array, transform = None, path_embeddings_df: pol.DataFrame = None):
        super().__init__()
        self.images             = images
        self.labels             = labels
        self.transform          = transform
        self.path_embeddings_df = path_embeddings_df

        self.features = [f"cifar_x{i}" for i in range(0, 2048)] + \
                        [f"tiny_imagenet_x{i}" for i in range(0, 2048)] + \
                        [f"imagenet_sketch_x{i}" for i in range(0, 2048)]

        if self.path_embeddings_df is not None:
            embeddings_df   = pol.read_csv(self.path_embeddings_df)
            self.embeddings = embeddings_df.to_numpy()

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
        
        if self.path_embeddings_df is not None:
            embedding = self.embeddings[idx, : -1]
            e_label   = self.embeddings[idx, -1]
            
            assert e_label ==  label
            return image, embedding, label
        else:
            return image, label     

class ImageNetDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform = None, path_embeddings_df: pol.DataFrame = None):
        super().__init__()
        self.data      = data
        self.transform = transform
        self.path_embeddings_df = path_embeddings_df

        self.features = [f"cifar_x{i}" for i in range(0, 2048)] + \
                        [f"tiny_imagenet_x{i}" for i in range(0, 2048)] + \
                        [f"imagenet_sketch_x{i}" for i in range(0, 2048)]

        if self.path_embeddings_df is not None:
            embeddings_df   = pol.read_csv(self.path_embeddings_df)
            self.embeddings = embeddings_df.to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data["path"].values[idx]

        image = cv2.imread(image_path)
        label = self.data["class"].values[idx]

        if self.transform:
            image = self.transform(image)
        
        if self.path_embeddings_df is not None:
            embedding = self.embeddings[idx, : -1]
            e_label   = self.embeddings[idx, -1]
            
            assert e_label ==  label
            return image, embedding, label
        else:
            return image, label  
