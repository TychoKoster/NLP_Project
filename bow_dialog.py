import read_data as rd
import numpy as np
import sklearn
from collections import Counter
import pickle
import string

if __name__ == "__main__":
	word_cnt = Counter()
	visual_feat_mapping, data = rd.read_data()
	translator = str.maketrans('', '', string.punctuation)
	for sample in data.values():
		caption = sample['caption']
		dialog = sample['dialog']

		# To add dialog as a feature.
		# for word in caption.split():
		# 	word_cnt[word] += 1
		for sentence in dialog:
			sentence = sentence[0].translate(translator) 
			for word in sentence.split():
				word_cnt[word] += 1
	word_list = list(word_cnt.keys())
	all_bow_dialogs = []
	for sample in data.values():
		caption = sample['caption']
		dialog = sample['dialog']
		dialog_list = []
		for sentence in dialog:
			sentence_vec = np.zeros(len(word_list))
			sentence = sentence[0].translate(translator)
			for word in sentence.split():
				word_index = word_list.index(word)
				sentence_vec[word_index] += 1
			dialog_list.append(sentence_vec)
		all_bow_dialogs.append(dialog)


	# for sample in data.values():
	# 	caption = sample['caption']
	# 	dialog = sample['dialog']
	# 	img_list = sample['img_list']
	# 	target_id = sample['target_img_id']
	# 	bow_dialog = []
	# 	# To add dialog as a feature.
	# 	# bow_caption = []
	# 	# for word in dialog.split():
	# 	# 	bow_caption.append(word_list.index(word))
	# 	for sentence in dialog:
	# 		bow_sentence = []
	# 		for word in sentence[0].split():
	# 			bow_sentence.append(word_list.index(word))
	# 		bow_dialog.append(bow_sentence)
	# 	# To add dialog as a feature.
	# 	# all_bow_captions.append(bow_caption)
	# 	all_bow_dialogs.append(bow_dialog)
	# with open("bow_dialog.txt", "wb") as fp:   #Pickling
	# 	pickle.dump(all_bow_dialogs, fp)

