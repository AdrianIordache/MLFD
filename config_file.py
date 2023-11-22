from common import *

PRINT_FREQ = 10
USE_WANDB  = True
RUN_NAME   = "exp-3-stoch-0.4"

config = dict(
	advanced  = True,
	dataset   = "TinyImageNet", # ["CIFAR100", "TinyImageNet", "ImageNetSketch"]
	
	image_size  = 256,
	optimizer   = "AdamW",
	scheduler   = "OneCycleLR", # "OneCycleLR",
	epochs      = 100,
	n_embedding = 768,
	activation  = nn.Identity(),

	expert = dict(
		model_name     = None, # "resnet18", tf_efficientnet_b0.in1k, "seresnext26t_32x4d"
		num_classes    = None, # 100, 200, 1000
		pretrained     = False, 
		drop_rate      = 0.4,
		drop_path_rate = 0.4, # stochastic depth
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

	label_smoothing  = 0.1,
	optimizer_param  = dict(
		lr           = 1e-3,
		weight_decay = 0.05,
	),

	scheduler_param  = dict(
		max_lr           = None,
		pct_start        = 0.133,
		div_factor       = 25,
		epochs           = None,
		steps_per_epoch  = 391,
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
		batch_size     = 256,
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

	config["expert"]["num_classes"]       = 100
	config["linear_teacher"]["n_outputs"] = 100
	config["conv_teacher"]["n_outputs"]   = 100

	if config["advanced"] == False:
		config["expert"]["model_name"] = "resnet18"
	else:
		config["expert"]["model_name"] = "convnextv2_tiny.fcmae_ft_in22k_in1k"

elif config["dataset"] == "TinyImageNet": 

	config["expert"]["num_classes"]       = 200
	config["linear_teacher"]["n_outputs"] = 200
	config["conv_teacher"]["n_outputs"]   = 200

	if config["advanced"] == False:
		config["expert"]["model_name"] = "tf_efficientnet_b0.in1k"
	else:
		config["expert"]["model_name"] = "swinv2_tiny_window8_256.ms_in1k"

else: 
	config["expert"]["num_classes"]       = 1000
	config["linear_teacher"]["n_outputs"] = 1000
	config["conv_teacher"]["n_outputs"]   = 1000

	if config["advanced"] == False:
		config["expert"]["model_name"] = "seresnext26t_32x4d"
	else:
		config["expert"]["model_name"] = "fastvit_sa24.apple_in1k"


config["mixup_param"]["num_classes"] = config["expert"]["num_classes"] 
config["mixup_param"]["label_smoothing"] = config["label_smoothing"]