from common import *

PRINT_FREQ = 10
USE_WANDB  = False
RUN_NAME   = "exp-15-multi-dist-base"

config = dict(
	dataset   = "ImageNetSketch", # ["CIFAR100", "TinyImageNet", "ImageNetSketch"]

	image_size  = 224,
	optimizer   = "Adam",
	scheduler   = None, # "OneCycleLR",
	epochs      = 200,
	n_embedding = 2048,
	activation  = nn.Identity(),

	expert = dict(
		model_name     = None, # "resnet18", tf_efficientnet_b0.in1k, "seresnext26t_32x4d"
		num_classes    = None, # 100, 200, 1000
		pretrained     = False, 
		drop_rate      = 0.0, # 0.3,
		drop_path_rate = 0.0, # 0.2
	),

    linear_teacher = dict(
    	n_inputs   = 2048 * 3,
    	n_outputs  = None,
    	p_dropout  = 0.9,
    ),

    conv_teacher = dict(
		nf_space   = 4096,
		nf_outputs = None,
		n_outputs  = None,

		nf_c = 512,
		nf_t = 1280,
		nf_s = 2048
	),

	optimizer_param  = dict(
		lr           = 0.0001,
		weight_decay = 0.001,
	),

	scheduler_param  = dict(
		max_lr           = None,
		pct_start        = 0.3,
		div_factor       = 25,
		epochs           = None,
		steps_per_epoch  = 66,
		final_div_factor = 1e4
	),

	train_transforms = [
		T.AutoAugment(policy = AutoAugmentPolicy.IMAGENET)
	],

	test_transforms = [

	],

	trainloader = dict(
		batch_size     = 192,
		shuffle        = False, 
		num_workers    = 2,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),
	
	testloader = dict(
		batch_size     = None,
		shuffle        = False, 
		num_workers    = 2,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),

	distillation_loss = AdaptedCELoss(),
	temperature       = 2,

    feature_map_loss  = nn.MSELoss(),
    kd_coefs          = [0.6, 0.2, 0.2],
    
)

config["scheduler_param"]["max_lr"] = config["optimizer_param"]["lr"]
config["scheduler_param"]["epochs"] = config["epochs"]
config["testloader"]["batch_size"]  = config["trainloader"]["batch_size"] * 2

assert config["dataset"] in ["CIFAR100", "TinyImageNet", "ImageNetSketch"]

if config["dataset"] == "CIFAR100": 
	config["expert"]["model_name"]        = "resnet18"
	config["expert"]["num_classes"]       = 100
	config["expert"]["drop_path_rate"]    = 0.0
	config["expert"]["drop_rate"]         = 0.0
	
	config["linear_teacher"]["n_outputs"] = 100
	config["conv_teacher"]["nf_outputs"]  = 512
	config["conv_teacher"]["n_outputs"]   = 100

elif config["dataset"] == "TinyImageNet": 
	config["expert"]["model_name"]        = "tf_efficientnet_b0.in1k"
	config["expert"]["num_classes"]       = 200
	config["expert"]["drop_path_rate"]    = 0.2
	config["expert"]["drop_rate"]         = 0.2
	
	config["linear_teacher"]["n_outputs"] = 200
	config["conv_teacher"]["nf_outputs"]  = 1280
	config["conv_teacher"]["n_outputs"]   = 200

else: 
	config["expert"]["model_name"]        = "seresnext26t_32x4d"
	config["expert"]["num_classes"]       = 1000
	config["expert"]["drop_path_rate"]    = 0.0
	config["expert"]["drop_rate"]         = 0.0
	
	config["linear_teacher"]["n_outputs"] = 1000
	config["conv_teacher"]["nf_outputs"]  = 2048
	config["conv_teacher"]["n_outputs"]   = 1000