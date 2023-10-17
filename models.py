from common import *

class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = timm.create_model(**config)

        if "efficientnet" in config["model_name"]:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif "resnet" in config["model_name"] or "resnext" in config["model_name"]:       
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()

        self.head = nn.Linear(in_features, config["num_classes"])
        
    def forward(self, x, embeddings = False, debug = False):
        x = self.model(x)
        if embeddings: return x
        x = self.head(x)
        return x


class IntermediateModel(nn.Module):
    def __init__(self, config, n_embedding, activation):
        super().__init__()
        self.model = timm.create_model(**config)
    
        if "efficientnet" in config["model_name"]:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif "resnet" in config["model_name"] or "resnext" in config["model_name"]:       
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            in_features = self.model.head.in_features
            self.model.head = nn.Identity()

        self.activation = activation
        self.intermediate = nn.Linear(in_features,    n_embedding)
        self.head  = nn.Linear(n_embedding, config["num_classes"])
        
    def forward(self, x, backbone = False, embeddings = False, debug = False):
        x = self.model(x)
        if backbone: return x
        x = self.activation(self.intermediate(x))
        if embeddings: return x
        x = self.head(x)
        return x