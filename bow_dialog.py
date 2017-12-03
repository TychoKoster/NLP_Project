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
import time

CONTEXT_SIZE = 1
EMBEDDING_DIM = 10
EPOCHS = 10
MODEL_PATH = 'CBOW.pt'

class CBOW(nn.Module):
	def __init__(self, context_size, embedding_dim, vocab_size):
		super(CBOW, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.linear = nn.Linear(embedding_dim, vocab_size)

	# Minimize (A*sum(q) + b), a linear combination with the sum of the embedded input
	def forward(self, inputs):
		# Embed the context of a word and sum it into an embedded vector
		embedded_input = self.embedding(inputs)
		embedded = embedded_input.sum(0)
		# Reshape to prevent input error when measuring loss
		embedded = embedded.view(1,-1)
		# Return the log probabilities
		return F.log_softmax(self.linear(embedded))

# Simple timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)

        print('The function ran for {0} seconds'.format(time.time() - start))
        return output
    return wrapper

def retrieve_context_data(context_data, vocabulary, sentence):
	split_sentence = sentence.split()
	vocabulary.extend([word for word in sentence.split()])
	for i in range(CONTEXT_SIZE, len(split_sentence) - CONTEXT_SIZE):
		context_lhs = split_sentence[i-CONTEXT_SIZE:i]
		context_rhs = split_sentence[i+1:i+1+CONTEXT_SIZE]
		context = np.concatenate((context_lhs, context_rhs))
		target = split_sentence[i]
		context_data.append((context, target))
	return context_data, vocabulary

# Retrieve context data and the vocabulary of the dialog and captions
def process_data(data):
	context_data = []
	vocabulary = []
	# translator = str.maketrans('', '', string.punctuation)
	# word_cnt = Counter([word for sample in data.values() for sentence in sample['dialog'] for word in sentence[0].split()])
	for sample in data.values():
		dialog = sample['dialog']
		for sentence in dialog:
			context_data, vocabulary = retrieve_context_data(context_data, vocabulary, sentence[0])
		caption = sample['caption']
		context_data, vocabulary = retrieve_context_data(context_data, vocabulary, caption)

	vocabulary = set(vocabulary)
	vocab_size = len(vocabulary)

	w2i = {word: i for i, word in enumerate(vocabulary)}

	return context_data, vocab_size, w2i


def context_to_index(context, w2i):
	indices = [w2i[word] for word in context]
	context_vector = autograd.Variable(torch.LongTensor(indices))
	return context_vector

@timer
def train_model(context_data, vocab_size, w2i):
	torch.manual_seed(1)
	losses = []
	# CBOW minimizes the negative log likelihood
	loss_function = nn.NLLLoss()
	model = CBOW(CONTEXT_SIZE, EMBEDDING_DIM, vocab_size)
	optimizer = optim.SGD(model.parameters(), lr=0.001)

	print(len(context_data))

	for iteration in range(EPOCHS):
		total_loss = torch.Tensor([0])
		for context, target in context_data:
			context_vector = context_to_index(context, w2i)
			model.zero_grad()
			probabilities = model(context_vector)
			target = autograd.Variable(torch.LongTensor([w2i[target]]))
			loss = loss_function(probabilities, target)
			loss.backward()
			optimizer.step()
			total_loss += loss.data
		losses.append(total_loss)
		print("Total loss epoch {0}: {1}".format(iteration, total_loss))

	return model

def save_model(model, path):
	torch.save(model, path)

def load_model(path):
	model = torch.load(path)
	return model

def main():
	_, data_easy, data_hard = rd.read_data()
	context_data, vocab_size, w2i = process_data(data_easy)
	model = train_model(context_data, vocab_size, w2i)
	save_model(model, MODEL_PATH)
	# model = load_model(MODEL_PATH)

if __name__ == "__main__":
	main()


