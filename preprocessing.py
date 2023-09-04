from common import *
from labels import *

def load_cifar(path):
    with open(path, "rb") as file:
        data_dict = pickle.load(file, encoding = "bytes")

    images = data_dict[b'data']
    labels = np.array(data_dict[b'fine_labels'])
    return images, labels

def compute_mean_and_std(train_images, test_images):
    cifar_images = np.vstack((train_images, test_images))

    mean = [
        np.mean(cifar_images[:,      : 1024], axis = (0, 1)) / 255,    
        np.mean(cifar_images[:, 1024 : 2048], axis = (0, 1)) / 255,
        np.mean(cifar_images[:, 2048 :     ], axis = (0, 1)) / 255
    ]

    std  = [
        np.std(cifar_images[:,      : 1024], axis = (0, 1)) / 255,    
        np.std(cifar_images[:, 1024 : 2048], axis = (0, 1)) / 255,
        np.std(cifar_images[:, 2048 :     ], axis = (0, 1)) / 255
    ]

    return mean, std

def prep_tiny_imagenet():
	PATH_TO_TINY_IMAGENET_TRAIN  = "./data/tiny-imagenet-200/train"
	PATH_TO_TINY_IMAGENET_VALID  = "./data/tiny-imagenet-200/val"
	PATH_TO_TINY_IMAGENET_LABELS = "./data/tiny-imagenet-200/words.txt"

	# Generate train data csv
	lines = dict()
	with open(PATH_TO_TINY_IMAGENET_LABELS, "r") as handler:
		line = handler.readline()
		while line:
			label, text  = line.strip("\n").split("\t")
			lines[label] = text
			line  = handler.readline()

	label_to_idx    = {}
	train_label_map = {
		"path"  : [],
		"class" : [],
		"label"	: [],
		"text"	: [],
	}

	for idx, folder in enumerate(glob.glob(PATH_TO_TINY_IMAGENET_TRAIN + "/*")):
		label       = folder.split("/")[-1]
		description = lines[label]
		
		label_to_idx[label] = (idx, description) 
		for image_path in glob.glob(f"{folder}/images/*.JPEG"):
			train_label_map["path"].append(image_path)
			train_label_map["class"].append(idx)
			train_label_map["label"].append(label)
			train_label_map["text"].append(description)

	train_label_map = pd.DataFrame(train_label_map)
	train_label_map = train_label_map.sample(frac = 1, random_state = SEED)
	train_label_map.reset_index(drop = True, inplace = True)
	train_label_map.to_csv("./data/tiny-imagenet-200/train.csv", index = False)
	display(train_label_map)

	# Generate valid data csv
	valid_label_map = {
		"path"  : [],
		"class" : [],
		"label"	: [],
		"text"	: [],
	}

	with open(PATH_TO_TINY_IMAGENET_VALID + "/val_annotations.txt", "r") as handler:
		line = handler.readline()
		while line:
			name       = line.strip("\n").split("\t")[0]
			image_path = f"{PATH_TO_TINY_IMAGENET_VALID}/images/{name}"

			label = line.strip("\n").split("\t")[1]
			(class_id, desc) = label_to_idx[label]

			valid_label_map["path"].append(image_path)
			valid_label_map["class"].append(class_id)
			valid_label_map["label"].append(label)
			valid_label_map["text"].append(desc)

			line  = handler.readline()

	valid_label_map = pd.DataFrame(valid_label_map)
	valid_label_map = valid_label_map.sample(frac = 1, random_state = SEED)
	valid_label_map.reset_index(drop = True, inplace = True)
	valid_label_map.to_csv("./data/tiny-imagenet-200/valid.csv", index = False)
	display(valid_label_map)

def prep_imagenet_sketch():
	PATH_TO_IMAGENET_SKETCH_IMAGES = "./data/imagenet-sketch/images"

	label_map = {
		"path"  : [],
		"class" : [],
		"label"	: [],
		"text"	: [],
	}

	for class_idx, directory in enumerate(glob.glob(PATH_TO_IMAGENET_SKETCH_IMAGES + "/*")):
		label = directory.split("/")[-1] 
		text  = IMAGENET_LABELS[label]

		for image_path in glob.glob(directory + "/*"):
			label_map["path"].append(image_path)
			label_map["class"].append(class_idx)
			label_map["label"].append(label)
			label_map["text"].append(text)


	label_map = pd.DataFrame(label_map)
	label_map = label_map.sample(frac = 1, random_state = SEED)
	label_map.reset_index(drop = True, inplace = True)
	label_map = generate_folds(label_map, "class")
	# label_map["train"] = label_map["fold"].apply(lambda x: True if x < 4 else False)

	train_data = label_map[label_map["fold"]  < 4].reset_index(drop = True, inplace = False)
	valid_data = label_map[label_map["fold"] == 4].reset_index(drop = True, inplace = False)
	
	train_data.to_csv("./data/imagenet-sketch/train.csv", index = False)
	valid_data.to_csv("./data/imagenet-sketch/valid.csv", index = False)
	
	display(train_data)
	display(valid_data)

if __name__ == "__main__":
	# prep_tiny_imagenet()
	# prep_imagenet_sketch()
	pass