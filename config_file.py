from common import *

PRINT_FREQ = 10
USE_WANDB  = True
RUN_NAME   = "exp-26-teacher-28"

config = dict(
	e_size    = 28,
	advanced  = False,
	dataset   = "CIFAR100", # ["CIFAR100", "TinyImageNet", "ImageNetSketch"]
	
	image_size  = 224,
	optimizer   = "AdamW",
	scheduler   = "OneCycleLR", # "OneCycleLR",
	epochs      = 137,
	n_embedding = 768,
	activation  = nn.Identity(),

	expert = dict(
		model_name     = None, # "resnet18", tf_efficientnet_b0.in1k, "seresnext26t_32x4d"
		num_classes    = None, # 100, 200, 1000
		pretrained     = False, 
		drop_rate      = 0.2,
		drop_path_rate = 0.3, # stochastic depth
	),

    teacher = dict(
    	nf_space   = 512,
    	nf_outputs = [128, 256, 512],  # [128, 256, 512],
		n_outputs  = None,

		activation = nn.GELU(),
		p_dropout  = 0.0
	),

	n_features_maps = dict(
		nf_c =  128, # 128, # 256,  # 512,  # 2048,
		nf_t =   40, #  40, # 112,  # 1280, # 2048,
		nf_s =  512, # 512, # 1024, # 2048, # 2048,
	),

	label_smoothing  = 0.1,
	optimizer_param  = dict(
		lr           = 0.00002,
		weight_decay = 0.05,
	),

	scheduler_param  = dict(
		max_lr           = None,
		pct_start        = 0.133,
		div_factor       = 25,
		epochs           = None,
		steps_per_epoch  = 99,
		final_div_factor = 1e4
	),

	use_timm = False,
	rand_aug = "rand-m3-n1-mstd0.5",
	re_prob  = 0.0,
	re_mode  = 'const',
	re_count = 1,

	train_transforms = [
		T.AutoAugment(policy = AutoAugmentPolicy.IMAGENET)
	],

	test_transforms = [

	],

	use_mixup   = False,
	mixup_param = dict(
		mixup_alpha     = 0.8, 
		cutmix_alpha    = 1.0, 
		prob            = 1.0, 
		switch_prob     = 0.0, 
		mode            = "batch",
		label_smoothing = None, 
		num_classes     = None
	),

	trainloader = dict(
		batch_size     = 512,
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

	config["teacher"]["n_outputs"]  = 100
	config["expert"]["num_classes"] = 100

	if config["advanced"] == False:
		config["expert"]["model_name"] = "resnet18"
	else:
		config["expert"]["model_name"] = "convnextv2_tiny.fcmae_ft_in22k_in1k"

elif config["dataset"] == "TinyImageNet": 

	config["teacher"]["n_outputs"]  = 200
	config["expert"]["num_classes"] = 200

	if config["advanced"] == False:
		config["expert"]["model_name"] = "tf_efficientnet_b0.in1k"
	else:
		config["expert"]["model_name"] = "swinv2_tiny_window8_256.ms_in1k"

else: 

	config["teacher"]["n_outputs"]  = 1000
	config["expert"]["num_classes"] = 1000

	if config["advanced"] == False:
		config["expert"]["model_name"] = "seresnext26t_32x4d"
	else:
		config["expert"]["model_name"] = "fastvit_sa24.apple_in1k"


config["mixup_param"]["num_classes"] = config["expert"]["num_classes"] 
config["mixup_param"]["label_smoothing"] = config["label_smoothing"]