from common import *

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

if __name__ == "__main__":
	prep_tiny_imagenet()