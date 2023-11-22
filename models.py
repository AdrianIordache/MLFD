from common import *
from timm.models.convnext import ConvNeXtBlock

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


class ExpertModel(nn.Module):
    def __init__(self, config, n_embedding, activation):
        super().__init__()
        self.model = timm.create_model(**config)
        # print(self.model)
        if "efficientnet" in config["model_name"]:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif "resnet" in config["model_name"] or "resnext" in config["model_name"]:       
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        else:
            self.in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()

        self.activation  = activation
        self.n_embedding = n_embedding

        self.latent_proj = nn.Linear(self.in_features, self.n_embedding)
        self.output_proj = nn.Linear(self.n_embedding, config["num_classes"])
        
    def forward(self, x):
        x = self.model(x)
        
        if self.n_embedding == self.in_features:
            x = self.output_proj(x)
        else:
            x = self.activation(self.latent_proj(x))
            x = self.output_proj(x)

        return x

class LinearTeacherModel(nn.Module):
    def __init__(self, config):
        super(LinearTeacherModel, self).__init__()
        self.config = config

        self.dropout    = nn.Dropout(p = config["p_dropout"])
        self.projection = nn.Linear(config["n_in"], config["n_out"]) 

    def forward(self, x):
        x = self.projection(self.dropout(x))
        return x
        
class ConvTeacherModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_bn = nn.BatchNorm2d(config["nf_c"])
        self.t_bn = nn.BatchNorm2d(config["nf_t"])
        self.s_bn = nn.BatchNorm2d(config["nf_s"])

        self.c_projection = nn.Conv2d(config["nf_c"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.t_projection = nn.Conv2d(config["nf_t"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.s_projection = nn.Conv2d(config["nf_s"], config["nf_space"], kernel_size = (1, 1), stride = 1)

        self.concat_norm       = nn.BatchNorm2d(config["nf_space"] * 3)
        self.filter_projection = nn.Conv2d(
            config["nf_space"] * 3, config["nf_outputs"], 
            kernel_size = (1, 1), stride = 1
        )
        self.projection_norm   = nn.BatchNorm2d(config["nf_outputs"])

        self.activation = nn.GELU()
        self.pooling    = nn.AdaptiveAvgPool2d(1)
        self.flatten    = nn.Flatten()

        self.dropout    = nn.Dropout()
        self.linear     = nn.Linear(config["nf_outputs"], config["n_outputs"])

        self.first_boundary  = config["nf_c"]
        self.second_boundary = config["nf_c"] + config["nf_t"]

    def forward(self, inputs, out_maps = False):
        x = inputs[:,                      :  self.first_boundary, :, :]
        y = inputs[:, self.first_boundary  : self.second_boundary, :, :]
        z = inputs[:, self.second_boundary :,                      :, :]

        x = self.c_projection(self.c_bn(x))
        y = self.t_projection(self.t_bn(y))
        z = self.s_projection(self.s_bn(z))

        out = torch.cat((x, y, z), dim = 1)
        out = self.activation(self.concat_norm(out))

        out  = self.filter_projection(out)
        maps = self.activation(self.projection_norm(out))

        out = self.flatten(self.pooling(maps))
        out = self.linear(self.dropout(out))

        if out_maps:
            return out, maps
        else:
            return out
        

