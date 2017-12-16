from collections import Counter
from itertools import compress
from itertools import chain
from nltk.corpus import stopwords
from stemming.porter2 import stem

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



import read_data as rd
import numpy as np
import collections
import sklearn
import pickle
import string
import time

CONTEXT_SIZE = 1
EMBEDDING_DIM = 300
ITERATIONS = 10
BATCH_SIZE = 256
MODEL_PATH_EASY = 'CBOW_EASY.pth'
MODEL_PATH_HARD = 'CBOW_HARD.pth'
EASY = 'Easy'
HARD = 'Hard' 
PREPROCCES = False

class CBOW(nn.Module):
    def __init__(self, embedding_dim, vocab_size, w2i):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.loss_function = nn.NLLLoss()
        self.w2i = w2i
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    # Minimize (A*sum(q) + b), a linear combination with the sum of the embedded input
    def forward(self, inputs, target):
        # Embed the context of a word and sum it into an embedded vector
        embedded_input = self.embedding(inputs)
        sum_embedding = torch.sum(embedded_input, dim=1)
        output = self.linear(sum_embedding)
        return self.compute_loss(F.log_softmax(output), target)

    def compute_loss(self, probabilities, target):
        return self.loss_function(probabilities, target)

    def embed_word_vector(self, word_vector):
        embedded_vector = self.embedding(autograd.Variable(torch.LongTensor(np.array(context_to_index(word_vector, self.w2i))).cuda())).cuda()
        embedded_vector = torch.sum(embedded_vector, dim=0)
        return embedded_vector.data.cpu().numpy()

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
# def process_data(data):
#     context_data = []
#     vocabulary = []
#     for sample in data:
#         dialog = sample['dialog']
#         for sentence in dialog:
#             context_data, vocabulary = retrieve_context_data(context_data, vocabulary, sentence[0])
#         caption = sample['caption']
#         context_data, vocabulary = retrieve_context_data(context_data, vocabulary, caption)
#     vocabulary.extend(['UNKNOWN'])
#     vocabulary = set(vocabulary)
#     vocab_size = len(vocabulary)

#     w2i = {word: i for i, word in enumerate(vocabulary)}

#     return context_data, vocab_size, w2i

def get_filtered_sentence(sentence):
    sentence = sentence.translate(str.maketrans("","", string.punctuation))
    filtered_sentence_list = [stem(word) for word in sentence.split() if word not in stopwords.words('english') if not word.isdigit()]
    filtered_sentence = " ".join(filtered_sentence_list)
    return filtered_sentence

def process_data(data):
    context_data = []
    vocabulary = []
    for sample in data:
        dialog = sample['dialog']
        for sentence in dialog:
            if PREPROCCES:
                sentence = get_filtered_sentence(sentence[0])
            else:
                sentence = sentence[0]
            context_data, vocabulary = retrieve_context_data(context_data, vocabulary, sentence)
        caption = sample['caption']
        if PREPROCCES:
            caption = get_filtered_sentence(caption)
        context_data, vocabulary = retrieve_context_data(context_data, vocabulary, caption)
    vocabulary.extend(['UNKNOWN'])
    vocabulary = set(vocabulary)
    vocab_size = len(vocabulary)

    w2i = {word: i for i, word in enumerate(vocabulary)}

    return context_data, vocab_size, w2i


def context_to_index(context, w2i):
    index_vec = []
    for word in context:
        if word in w2i.keys():
            index_vec.append(w2i[word])
        else:
            index_vec.append(w2i['UNKNOWN'])
    return index_vec

def save_loss(loss):
    writeline = "{0} \n".format(loss)
    if PREPROCCES:
        with open("PREPROCCES_HARD_Total_loss.txt", "a+") as f:
            f.write(writeline)
    else:
        with open("HARD_Total_loss.txt", "a+") as f:
            f.write(writeline)        

@timer
def train_model_batches(context_data, vocab_size, w2i):
    torch.manual_seed(1)

    # CBOW minimizes the negative log likelihood
    model = CBOW(EMBEDDING_DIM, vocab_size, w2i)
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # context_data = context_data[:2000]
    for iteration in range(ITERATIONS):
        total_loss = torch.Tensor([0]).cuda()
        for i in range(0, len(context_data), BATCH_SIZE):
            if i + BATCH_SIZE >= len(context_data):
                batch = context_data[i:]
            else:
                batch = context_data[i:i+BATCH_SIZE]

            context_vector = np.array([context_to_index(data_point[0], w2i) for data_point in batch])
            target = np.array([np.array([w2i[data_point[1]]]) for data_point in batch])
            
            context_vector = autograd.Variable(torch.LongTensor(context_vector)).cuda()
            target = autograd.Variable(torch.LongTensor(target)).squeeze().cuda()

            model.zero_grad()
            optimizer.zero_grad()

            loss = model(context_vector, target)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
        save_loss(total_loss)
        print("Total loss iteration {0}: {1}".format(iteration, total_loss))

    return model

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model

def main():
    _, _, train_data, val_data, test_data = rd.read_data(HARD)
    data = list(train_data.values())
    data.extend(list(val_data.values()))
    data.extend(list(test_data.values()))
    context_data, _, _ = process_data(list(train_data.values()))
    _, vocab_size, w2i = process_data(data)
    print("Preproccessed")

    model = train_model_batches(context_data, vocab_size, w2i)
    if PREPROCCES:
        path = "PREPROCCESS_" + MODEL_PATH_HARD
        print(path)
        save_model(model, path)
    else:
        save_model(model, MODEL_PATH_HARD)
    # model = load_model(MODEL_PATH_HARD)
    # print(model.embed_word_vector(['yachts', 'hello']))
    # print(model.embed_index_vector([100]))


if __name__ == "__main__":
    main()


