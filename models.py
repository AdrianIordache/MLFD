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
        # if embeddings: return x
        logits = self.head(x)

        if embeddings:
            return x, logits
            
        return x

class ExpertModel(nn.Module):
    def __init__(self, config, n_embedding, activation):
        super().__init__()
        self.model = timm.create_model(**config)
        # print(self.model)
        if "efficientnet" in config["model_name"]:
            self.in_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif "resnet" in config["model_name"] or "resnext" in config["model_name"]:       
            self.in_features = self.model.fc.in_features
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
            self.model.head.fc = nn.Identity()

        self.CIFAR100Head       = nn.Linear(in_features, 100)
        self.TinyImageNetHead   = nn.Linear(in_features, 200)
        self.ImageNetSketchHead = nn.Linear(in_features, 1000)
        
    def forward(self, x, head):
        assert head in ["CIFAR100", "TinyImageNet", "ImageNetSketch"]

        features = self.model(x)

        if head == "CIFAR100":
            return self.CIFAR100Head(features)

        if head == "TinyImageNet":
            return self.TinyImageNetHead(features)

        if head == "ImageNetSketch":
            return self.ImageNetSketchHead(features) 

        return features


class BaselineExtendedModel(nn.Module):
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
            self.model.head.fc = nn.Identity()

        self.extended_head  = nn.Linear(in_features, 100 + 200 + 1000)
        
    def forward(self, x):
        features = self.model(x)
        x = self.extended_head(features)

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
        
##################################################################################################################################################

class S1Teacher(nn.Module):
    def __init__(self, config):
        super(S1Teacher, self).__init__()
        self.config = config

        assert len(config["nf_outputs"]) == 0, f"S1: Wrong teacher, len(config['nf_outputs']) = {len(config['nf_outputs'])}"

        n_inputs = config["nf_c"] + config["nf_t"] + config["nf_s"]

        self.dropout    = nn.Dropout(p = config["p_dropout"])
        self.projection = nn.Linear(n_inputs, config["n_outputs"]) 

    def forward(self, x):
        x = self.projection(self.dropout(x))
        return x
        
class S1TeacherConv(nn.Module):
    def __init__(self, config):
        super(S1Teacher, self).__init__()
        self.config = config

        assert len(config["nf_outputs"]) == 0, f"S1: Wrong teacher, len(config['nf_outputs']) = {len(config['nf_outputs'])}"

        n_inputs = config["nf_c"] + config["nf_t"] + config["nf_s"]


        self.dropout     = nn.Dropout(p = config["p_dropout"])
        self.projection  = nn.Conv1d(1,   2048, kernel_size = 768, stride = 768) 
        self.projection2 = nn.Linear(2048, 100) 
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.activation = config["activation"]

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.activation(self.projection(self.dropout(x)))
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.projection2(x)
        # x = x.squeeze(2)
        return x

class S1TeacherAdv(nn.Module):
    def __init__(self, config):
        super(S1Teacher, self).__init__()
        self.config = config

        assert len(config["nf_outputs"]) == 0, f"S1: Wrong teacher, len(config['nf_outputs']) = {len(config['nf_outputs'])}"

        n_inputs = config["nf_c"] + config["nf_t"] + config["nf_s"]

        self.dropout  = nn.Dropout(p = config["p_dropout"])

        self.c_bn = nn.BatchNorm1d(config["nf_c"])
        self.t_bn = nn.BatchNorm1d(config["nf_t"])
        self.s_bn = nn.BatchNorm1d(config["nf_s"])

        self.c_filter = nn.Linear(config["nf_c"], config["nf_space"])
        self.t_filter = nn.Linear(config["nf_t"], config["nf_space"])
        self.s_filter = nn.Linear(config["nf_s"], config["nf_space"])

        self.projection_norm = nn.BatchNorm1d(config["nf_space"] * 3)
        self.projection = nn.Linear(config["nf_space"] * 3, config["n_outputs"]) 
        
    def forward(self, inputs):
        x = inputs[:,      : 768 ]
        y = inputs[:, 768  : 1536]
        z = inputs[:, 1536 :     ]

        x = self.c_filter(self.c_bn(x))
        y = self.t_filter(self.t_bn(y))
        z = self.s_filter(self.s_bn(z))

        out = torch.cat((x, y, z), dim = 1)
        out = self.activation(self.projection_norm(out))
        out = self.projection(self.dropout(out))

        return out
        


class S7Teacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert len(config["nf_outputs"]) == 1, f"S7: Wrong teacher, len(config['nf_outputs']) = {len(config['nf_outputs'])}"

        self.c_bn = nn.BatchNorm2d(config["nf_c"])
        self.t_bn = nn.BatchNorm2d(config["nf_t"])
        self.s_bn = nn.BatchNorm2d(config["nf_s"])
        # self.o_bn = nn.BatchNorm2d(config["nf_o"])

        self.c_projection = nn.Conv2d(config["nf_c"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.t_projection = nn.Conv2d(config["nf_t"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.s_projection = nn.Conv2d(config["nf_s"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        # self.o_projection = nn.Conv2d(config["nf_o"], config["nf_space"], kernel_size = (1, 1), stride = 1)

        self.concat_norm       = nn.BatchNorm2d(config["nf_space"] * 3)
        self.filter_projection = nn.Conv2d(
            config["nf_space"] * 3, config["nf_outputs"][0], 
            kernel_size = (1, 1), stride = 1
        )
        self.projection_norm   = nn.BatchNorm2d(config["nf_outputs"][0])

        self.activation = config["activation"]
        self.pooling    = nn.AdaptiveAvgPool2d(1)
        self.flatten    = nn.Flatten()

        self.dropout    = nn.Dropout(config["p_dropout"])
        self.linear     = nn.Linear(config["nf_outputs"][0], config["n_outputs"])

        self.first_boundary  = config["nf_c"]
        self.second_boundary = config["nf_c"] + config["nf_t"]
        # self.third_boundary  = config["nf_c"] + config["nf_t"] + config["nf_s"]

    def forward(self, inputs, out_maps = False):
        x = inputs[:,                      : self.first_boundary,  :, :]
        y = inputs[:, self.first_boundary  : self.second_boundary,  :, :]
        z = inputs[:, self.second_boundary :,  :, :]
        # o = inputs[:, self.third_boundary  : ,  :, :]

        x = self.c_projection(self.c_bn(x))
        y = self.t_projection(self.t_bn(y))
        z = self.s_projection(self.s_bn(z))
        # o = self.o_projection(self.o_bn(o))

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


class S14Teacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert len(config["nf_outputs"]) == 2, f"S14: Wrong teacher, len(config['nf_outputs']) = {len(config['nf_outputs'])}"
        
        self.c_bn = nn.BatchNorm2d(config["nf_c"])
        self.t_bn = nn.BatchNorm2d(config["nf_t"])
        self.s_bn = nn.BatchNorm2d(config["nf_s"])

        self.c_projection = nn.Conv2d(config["nf_c"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.t_projection = nn.Conv2d(config["nf_t"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.s_projection = nn.Conv2d(config["nf_s"], config["nf_space"], kernel_size = (1, 1), stride = 1)

        self.concat_norm      = nn.BatchNorm2d(config["nf_space"] * 3)
        self.sized_projection = nn.Conv2d(
            config["nf_space"] * 3, config["nf_outputs"][0], 
            kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False
        )
        self.sized_norm  = nn.BatchNorm2d(config["nf_outputs"][0])

        self.filter_projection = nn.Conv2d(
            config["nf_outputs"][0], config["nf_outputs"][1], 
            kernel_size = (1, 1), stride = 1 
        )
        self.filter_norm = nn.BatchNorm2d(config["nf_outputs"][1])

        self.activation = config["activation"]
        self.pooling    = nn.AdaptiveAvgPool2d(1)
        self.flatten    = nn.Flatten()

        self.dropout    = nn.Dropout(config["p_dropout"])
        self.linear     = nn.Linear(config["nf_outputs"][1], config["n_outputs"])

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

        out    = self.sized_projection(out)
        maps_1 = self.activation(self.sized_norm(out))

        out    = self.filter_projection(maps_1)
        maps_2 = self.activation(self.filter_norm(out))

        out = self.flatten(self.pooling(maps_2))
        out = self.linear(self.dropout(out))
        
        if out_maps:
            return out, maps_1, maps_2
        else:
            return out

class S28Teacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert len(config["nf_outputs"]) == 3, f"S28: Wrong teacher, len(config['nf_outputs']) = {len(config['nf_outputs'])}"
        
        self.c_bn = nn.BatchNorm2d(config["nf_c"])
        self.t_bn = nn.BatchNorm2d(config["nf_t"])
        self.s_bn = nn.BatchNorm2d(config["nf_s"])

        self.c_projection = nn.Conv2d(config["nf_c"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.t_projection = nn.Conv2d(config["nf_t"], config["nf_space"], kernel_size = (1, 1), stride = 1)
        self.s_projection = nn.Conv2d(config["nf_s"], config["nf_space"], kernel_size = (1, 1), stride = 1)

        self.concat_norm      = nn.BatchNorm2d(config["nf_space"] * 3)
        self.upper_projection = nn.Conv2d(
            config["nf_space"] * 3, config["nf_outputs"][0], 
            kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False
        )
        self.upper_norm  = nn.BatchNorm2d(config["nf_outputs"][0])

        self.lower_projection = nn.Conv2d(
            config["nf_outputs"][0], config["nf_outputs"][1], 
            kernel_size = (3, 3), stride = (2, 2), padding = (1, 1), bias = False
        )
        self.lower_norm  = nn.BatchNorm2d(config["nf_outputs"][1])

        self.filter_projection = nn.Conv2d(
            config["nf_outputs"][1], config["nf_outputs"][2], 
            kernel_size = (1, 1), stride = 1 
        )
        self.filter_norm = nn.BatchNorm2d(config["nf_outputs"][2])

        self.activation = config["activation"]
        self.pooling    = nn.AdaptiveAvgPool2d(1)
        self.flatten    = nn.Flatten()

        self.dropout    = nn.Dropout(config["p_dropout"])
        self.linear     = nn.Linear(config["nf_outputs"][2], config["n_outputs"])

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

        out    = self.upper_projection(out)
        maps_1 = self.activation(self.upper_norm(out))

        out    = self.lower_projection(maps_1)
        maps_2 = self.activation(self.lower_norm(out))

        out    = self.filter_projection(maps_2)
        maps_3 = self.activation(self.filter_norm(out))

        out = self.flatten(self.pooling(maps_3))
        out = self.linear(self.dropout(out))
        
        if out_maps:
            return out, maps_1, maps_2, maps_3
        else:
            return out


            