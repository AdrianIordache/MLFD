from common import *

CFG = dict(
	image_size = 224,
	optimizer  = "Adam",
	scheduler  = "OneCycleLR",
	epochs     = 2,

	model = dict(
		model_name     = "resnet18",
		num_classes    = 100,
		pretrained     = False, 
		drop_rate      = 0.0, # 0.3,
		drop_path_rate = 0.0, # 0.2
	),

	optimizer_param  = dict(
		lr           = 0.001,
		weight_decay = 0.0001,
	),

	scheduler_param  = dict(
		max_lr           = None,
		pct_start        = 0.3,
		div_factor       = 25,
		epochs           = None,
		steps_per_epoch  = 16,
		final_div_factor = 1e4
	),

	train_transforms = [
		# T.AutoAugment(policy = AutoAugmentPolicy.IMAGENET)
	],

	test_transforms = [

	],

	trainloader = dict(
		batch_size     = 64,
		shuffle        = False, 
		num_workers    = 4,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),
	
	testloader = dict(
		batch_size     = 64,
		shuffle        = False, 
		num_workers    = 4,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),

	print_freq = 1,
	use_wandb  = True,
)

CFG["scheduler_param"]["max_lr"] = CFG["optimizer_param"]["lr"]
CFG["scheduler_param"]["epochs"] = CFG["epochs"]