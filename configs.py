
# Dataset configurations

datasets = ["wider", "pa-100k", "ms-coco"]


WIDER_DATA_DIR = "/HGUO/WIDER_DATASET/Image"
WIDER_ANNO_DIR = "/HGUO/WIDER_DATASET/wider_attribute_annotation"

PA100K_DATA_dir = "/data/hguo/Datasets/PA-100K/release_data"
PA100K_ANNO_FILE = "/data/hguo/Datasets/PA-100K/annotation/annotation.mat"

MSCOCO_DATA_DIR = "/data/hguo/Datasets/MS-COCO"

# pre-calculated weights to balance positive and negative samples of each label
# as defined in Li et al. ACPR'15
# WIDER dataset
wider_pos_ratio = [0.5669, 0.2244, 0.0502, 0.2260, 0.2191, 0.4647, 0.0699, 0.1542, \
	0.0816, 0.3621, 0.1005, 0.0330, 0.2682, 0.0543]

# PA-100K dataset
pa100k_pos_ratio = [0.460444, 0.013456, 0.924378, 0.062167, 0.352667, 0.294622, \
	0.352711, 0.043544, 0.179978, 0.185000, 0.192733, 0.160100, 0.009522, \
	0.583400, 0.416600, 0.049478, 0.151044, 0.107756, 0.041911, 0.004722, \
	0.016889, 0.032411, 0.711711, 0.173444, 0.114844, 0.006000]

# MS-COCO dataset

def get_configs(dataset):
	opts = {}
	if not dataset in datasets:
		raise Exception("Not supported dataset!")
	else:
		if dataset == "wider":
			opts["dataset"] = "WIDER"
			opts["num_labels"] = 14
			opts["data_dir"] = WIDER_DATA_DIR
			opts["anno_dir"] = WIDER_ANNO_DIR
			opts["pos_ratio"] = wider_pos_ratio
		else:
			if dataset == "pa-100k":
				# will be added later
				pass
			else:
				pass
	return opts