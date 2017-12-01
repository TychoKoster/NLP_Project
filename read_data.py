import h5py
import json
import numpy as np

def read_data():
	# img_features = np.asarray(h5py.File('IR_image_features.h5', 'r')['img_features'])

	with open('IR_img_features2id.json', 'r') as f:
	     visual_feat_mapping = json.load(f)['IR_imgid2id']

	with open('Data/Easy/IR_train_easy.json') as f:
		data = json.load(f)
	return visual_feat_mapping, data
