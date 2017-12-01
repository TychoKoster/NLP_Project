from collections import Counter

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import read_data as rd
import numpy as np
import sklearn
import pickle
import string

CONTEXT_SIZE = 1
EPOCHS = 10

# https://gist.github.com/GavinXing/9954ea846072e115bb07d9758892382c
class CBOW(nn.Module):
	def __init__(self, context_size, embedding_size, vocabulary_size):
		super(CBOW, self).__init__()
		self.embedding = nn.Embedding(vocabulary_size, embedding_size)
		self.linear = nn.Linear(embedding_size, vocabulary_size)

	def forward(self, context):
		embedded_input = self.embedding(context)
		embedded = embedded_input.sum(dim=0)
		out = self.linear(embedded)
		out = F.log_softmax(out)
		return out


def make_context_vector(context, w2i):
	indices = [w2i[word] for word in context]
	context_vector = autograd.Variable(torch.LongTensor(indices))
	return context_vector

def main():
	_, data_easy, data_hard = rd.read_data()
	context_data = []
	# translator = str.maketrans('', '', string.punctuation)
	word_cnt = Counter([word for sample in data_easy.values() for sentence in sample['dialog'] for word in sentence[0].split()])
	for sample in data_easy.values():
		caption = sample['caption']
		dialog = sample['dialog']
		for sentence in dialog:
			split_sentence = sentence[0].split()
			for i in range(CONTEXT_SIZE, len(split_sentence) - CONTEXT_SIZE):
				context_lhs = split_sentence[i-CONTEXT_SIZE:i]
				context_rhs = split_sentence[i+1:i+1+CONTEXT_SIZE]
				context = np.concatenate((context_lhs, context_rhs))
				target = split_sentence[i]
				context_data.append((context, target))


	vocabulary_size = len(word_cnt)
	vocabulary = word_cnt.keys()
	w2i = {word: i for i, word in enumerate(vocabulary)}

	loss_function = nn.CrossEntropyLoss()
	model = CBOW(CONTEXT_SIZE, 10, vocabulary_size)
	optimizer = optim.SGD(model.parameters(), lr=0.001)

	for iteration in range(EPOCHS):
		total_loss = 0
		for context, target in context_data:
			context_vector = make_context_vector(context, w2i)
			model.zero_grad()
			probabilities = model(context_vector)
			# Fuck pytorch
			probabilities = probabilities.view(1, vocabulary_size)
			target = autograd.Variable(torch.LongTensor([w2i[target]]))
			loss = loss_function(probabilities, target)
			loss.backward()
			print(loss)
			optimizer.step()
			total_loss += loss.data
		print(total_loss)


if __name__ == "__main__":
	main()

	# for i in range(CONTEXT_SIZE, len(vocabulary) - CONTEXT_SIZE):
	# 	print(i)

	# word_list = list(word_cnt.keys())
	# all_bow_dialogs = []
	# for sample in data.values():
	# 	caption = sample['caption']
	# 	dialog = sample['dialog']
	# 	dialog_list = []
	# 	for sentence in dialog:
	# 		sentence_vec = np.zeros(len(word_list))
	# 		sentence = sentence[0].translate(translator)
	# 		for word in sentence.split():
	# 			word_index = word_list.index(word)
	# 			sentence_vec[word_index] += 1
	# 		dialog_list.append(sentence_vec)
	# 	all_bow_dialogs.append(dialog)


