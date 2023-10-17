from common import *

PRINT_FREQ = 10
USE_WANDB  = False

student_config = dict(
	image_size  = 224,
	optimizer   = "Adam",
	scheduler   = None, # "OneCycleLR",
	epochs      = 200,
	n_embedding = 2048,
	activation  = nn.Identity(),

	model = dict(
		model_name     = "tf_efficientnet_b0.in1k",
		num_classes    = 200, # 100,
		pretrained     = False, 
		drop_rate      = 0.2, # 0.3,
		drop_path_rate = 0.2, # 0.2
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
		batch_size     = 240,
		shuffle        = False, 
		num_workers    = 4,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),
	
	testloader = dict(
		batch_size     = None,
		shuffle        = False, 
		num_workers    = 4,
		pin_memory     = True,
		sampler        = sampler, 
		worker_init_fn = seed_worker, 
		drop_last      = False
	),

	distillation_loss = AdaptedCELoss(),
	temperature       = 2,
	kd_alpha          = 0.0
)

student_config["scheduler_param"]["max_lr"] = student_config["optimizer_param"]["lr"]
student_config["scheduler_param"]["epochs"] = student_config["epochs"]
student_config["testloader"]["batch_size"]  = student_config["trainloader"]["batch_size"] * 2

teacher_config = dict(
    epochs    = 2420,  
    dataset   = "ImageNetSketch", # ["CIFAR100", "TinyImageNet", "ImageNetSketch"]
    
	optimizer = "Adam",
    scheduler = None, # "OneCycleLR",
    
    model     = None,
    p_dropout = 0.9,
    n_outputs = None,

    optimizer_param  = dict(
        lr           = 0.0002,
        weight_decay = 0.000,
    ),

    scheduler_param  = dict(
        max_lr           = None,
        pct_start        = 0.3,
        div_factor       = 25,
        epochs           = None,
        steps_per_epoch  = 25,
        final_div_factor = 1e4
    ),

    trainloader = dict(
        batch_size     = 2048,
        shuffle        = False, 
        num_workers    = 28,
        pin_memory     = True,
        sampler        = sampler, 
        worker_init_fn = seed_worker, 
        drop_last      = False
    ),

    testloader = dict(
        batch_size     = None,
        shuffle        = False, 
        num_workers    = 28,
        pin_memory     = True,
        sampler        = sampler, 
        worker_init_fn = seed_worker, 
        drop_last      = False
    ),
)

teacher_config["scheduler_param"]["max_lr"]     = teacher_config["optimizer_param"]["lr"]
teacher_config["scheduler_param"]["epochs"]     = teacher_config["epochs"]
teacher_config["testloader"]["batch_size"]      = teacher_config["trainloader"]["batch_size"] * 4

assert teacher_config["dataset"] in ["CIFAR100", "TinyImageNet", "ImageNetSketch"]

if teacher_config["dataset"] == "CIFAR100": teacher_config["n_outputs"] = 100
elif teacher_config["dataset"] == "TinyImageNet": teacher_config["n_outputs"] = 200
else: teacher_config["n_outputs"] = 1000