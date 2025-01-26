from common import *

PRINT_FREQ = 10
USE_WANDB  = True
RUN_NAME   = "exp-3-student-grid2"

config = dict(
	e_size    = 7,
	advanced  = False,
	dataset   = "AfricanWildlife", # ["CIFAR100", "TinyImageNet", "ImageNetSketch", "Caltech101", "Flowers102", "CUB200", "OxfordPets"]
	
	image_size   = 384,
	optimizer    = "AdamW",
	scheduler    = None, # None, # "OneCycleLR",
	epochs       = 300,
	n_embedding  = 2048,
	activation   = nn.Identity(),
	accumulation = 1,

	expert = dict(
		model_name     = None, # "resnet18", tf_efficientnet_b0.in1k, "seresnext26t_32x4d"
		num_classes    = None, # 100, 200, 1000
		pretrained     = True, 
		drop_rate      = 0.0,
		drop_path_rate = 0.0, # stochastic depth
	),

	# expert_t = dict(
	# 	model_name     = "resnet18", # "resnet18", tf_efficientnet_b0.in1k, "seresnext26t_32x4d"
	# 	num_classes    = 100, # 100, 200, 1000
	# 	pretrained     = False, 
	# 	drop_rate      = 0.0,
	# 	drop_path_rate = 0.0, # stochastic depth
	# ),


    teacher = dict(
    	nf_space   = 2048 * 3,
    	nf_outputs = [2048],  # [40, 112, 1280], # [128, 256, 512],
		n_outputs  = 16, # None,

		activation = nn.GELU(),
		p_dropout  = 0.9
	),

	n_features_maps = dict(
		nf_c = 512,  # 768,  # 128, # 256,  # 512,  # 2048,
		nf_t = 1280, # 768, #  40, # 112,  # 1280, # 2048,
		nf_s = 2048, # 768, # 512, # 1024, # 2048, # 2048,
		# nf_o = 512,  # 128, # 256,  # 512,  # 2048
	),

	# label_smoothing  = 0.1,
	optimizer_param  = dict(
		lr           = 0.0001,
		weight_decay = 0.001
	),

	scheduler_param  = dict(
		max_lr           = None,
		pct_start        = 0.133,
		div_factor       = 25,
		epochs           = None,
		steps_per_epoch  = 13,
		final_div_factor = 10000 # 0.1 # 10000
	),

	# use_timm = False,
	# rand_aug = "rand-m3-n1-mstd0.5",
	# re_prob  = 0.0,
	# re_mode  = 'const',
	# re_count = 1,

	train_transforms = [
		T.AutoAugment(policy = AutoAugmentPolicy.IMAGENET)
	],

	test_transforms = [

	],

	# use_mixup   = False,
	# mixup_param = dict(
	# 	mixup_alpha     = 0.8, 
	# 	cutmix_alpha    = 1.0, 
	# 	prob            = 1.0, 
	# 	switch_prob     = 0.0, 
	# 	mode            = "batch",
	# 	label_smoothing = None, 
	# 	num_classes     = None
	# ),

	trainloader = dict(
		batch_size     = 104,
		shuffle        = False, 
		num_workers    = 0,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),
	
	testloader = dict(
		batch_size     = None,
		shuffle        = False, 
		num_workers    = 0,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),

	distillation_loss = AdaptedCELoss(),
	temperature       = 2,

    feature_map_loss  = nn.MSELoss(),
    kd_coefs          = [0.6, 0.2, 0.2],
    
	mixed_precision   = True,
	# lambda_intra      = 0.0008,
	# lambda_inter      = 0.0005,

	# norm_alpha        = 10,
	# norm_scale        = 8
	grid_size         = 2
)

config["scheduler_param"]["max_lr"] = config["optimizer_param"]["lr"]
config["scheduler_param"]["epochs"] = config["epochs"]
config["testloader"]["batch_size"]  = config["trainloader"]["batch_size"] * 2

config["scheduler_param"]["steps_per_epoch"] = config["scheduler_param"]["steps_per_epoch"] // config["accumulation"]

assert config["dataset"] in ["CIFAR100", "TinyImageNet", "ImageNetSketch", "Caltech101", "Flowers102", "CUB200", "OxfordPets", "VOC", "COCO", "AfricanWildlife"]

if config["dataset"] == "CIFAR100" or config["dataset"] == "Caltech101" or config["dataset"] == "OxfordPets": 

	config["teacher"]["n_outputs"]  = 100 # 37 # 102 # 102 # 100
	config["expert"]["num_classes"] = 100 #  37 # 102 # 102 # 100

	if config["advanced"] == False:
		config["expert"]["model_name"] = "resnet18" # "seresnext26t_32x4d" # "tf_efficientnet_b0.in1k" # "resnet18"
	else:
		config["expert"]["model_name"] = "convnextv2_tiny.fcmae_ft_in22k_in1k"

elif config["dataset"] == "TinyImageNet" or config["dataset"] == "Flowers102": 

	config["teacher"]["n_outputs"]  = 200 # 102 # 200
	config["expert"]["num_classes"] = 200 # 102 # 200

	if config["advanced"] == False:
		config["expert"]["model_name"] = "tf_efficientnet_b0.in1k" # "seresnext26t_32x4d" # "resnet18" # "tf_efficientnet_b0.in1k"
	else:
		config["expert"]["model_name"] = "swinv2_tiny_window8_256.ms_in1k"

elif config["dataset"] == "ImageNetSketch" or config["dataset"] == "CUB200":

	config["teacher"]["n_outputs"]  = 1000 # 200 # 1000
	config["expert"]["num_classes"] = 1000 # 200 # 1000

	if config["advanced"] == False:
		config["expert"]["model_name"] = "seresnext26t_32x4d" # "tf_efficientnet_b0.in1k" # "resnet18" # "seresnext26t_32x4d"
	else:
		config["expert"]["model_name"] = "fastvit_sa24.apple_in1k"

elif config["dataset"] == "VOC":
	config["expert"]["num_classes"] = 16 # 16 # 24 # 4 * 4 # 9 * 9
	config["expert"]["model_name"] = "resnet18" # swin_base_patch4_window12_384.ms_in22k_ft_in1k"

elif config["dataset"] == "COCO":
	config["expert"]["num_classes"] = 16 # 16 # 24 # 4 * 4 # 9 * 9
	config["expert"]["model_name"] = "tf_efficientnet_b0.in1k" # swin_base_patch4_window12_384.ms_in22k_ft_in1k"

elif config["dataset"] == "AfricanWildlife":
	config["expert"]["num_classes"] = 16 # 16 # 24 # 4 * 4 # 9 * 9
	config["expert"]["model_name"] = "seresnext26t_32x4d" # swin_base_patch4_window12_384.ms_in22k_ft_in1k"


# config["mixup_param"]["num_classes"] = config["expert"]["num_classes"] 
# config["mixup_param"]["label_smoothing"] = config["label_smoothing"]