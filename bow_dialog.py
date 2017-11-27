import read_data as rd
import numpy as np
import sklearn
from collections import Counter
import pickle

if __name__ == "__main__":
	word_cnt = Counter()
	visual_feat_mapping, data = rd.read_data()
	for sample in data.values():
		caption = sample['caption']
		dialog = sample['dialog']
		img_list = sample['img_list']
		target_id = sample['target_img_id']
		# To add dialog as a feature.
		# for word in dialog.split():
		# 	word_cnt[word] += 1
		for sentence in dialog:
			for word in sentence[0].split():
				word_cnt[word] += 1
	word_list = list(word_cnt.keys())
	all_bow_dialogs = []
	for sample in data.values():
		caption = sample['caption']
		dialog = sample['dialog']
		img_list = sample['img_list']
		target_id = sample['target_img_id']
		bow_dialog = []
		# To add dialog as a feature.
		# bow_caption = []
		# for word in dialog.split():
		# 	bow_caption.append(word_list.index(word))
		for sentence in dialog:
			bow_sentence = []
			for word in sentence[0].split():
				bow_sentence.append(word_list.index(word))
			bow_dialog.append(bow_sentence)
		# To add dialog as a feature.
		# all_bow_captions.append(bow_caption)
		all_bow_dialogs.append(bow_dialog)
	with open("bow_dialog.txt", "wb") as fp:   #Pickling
		pickle.dump(all_bow_dialogs, fp)

