import h5py
import json
import numpy as np


def read_data(level):
	data_path = 'Data/' + level + '/'
	train_file = data_path + 'IR_train_' + level.lower() + '.json'
	val_file = data_path + 'IR_val_' + level.lower() + '.json'
	test_file = data_path + 'IR_test_' + level.lower() + '.json'

	with open('IR_img_features2id.json', 'r') as f:
	    visual_feat_mapping = json.load(f)['IR_imgid2id']

	with open(train_file) as f:
		train_data = json.load(f)

	with open(val_file) as f:
		val_data = json.load(f)

	with open(test_file) as f:
		test_data = json.load(f)

	img_features = np.asarray(h5py.File('Data/IR_image_features.h5', 'r')['img_features'])

	return img_features, visual_feat_mapping, train_data, val_data, test_data
