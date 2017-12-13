# Shouldn't we use cross entropy as loss?
# Shuffle data?

from collections import Counter
from bow_dialog import CBOW

#from random import shuffle
from sklearn.utils import shuffle

import torch.cuda
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import read_data as rd
import numpy as np
import sklearn
import pickle
import string
import time

NUM_LAYERS = 1
CONTEXT_SIZE = 3
SEQ_LEN = CONTEXT_SIZE
EMBEDDING_DIM = 300
EPOCHS = 5
MODEL_PATH = 'GRULM.pth'
EMBED_MODEL_PATH = 'CBOW_EASY.pth'
BATCH_SIZE = 256
HIDDEN_SIZE = 128
LEARNING_RATE = 0.002

class GRULM(nn.Module):
    def __init__(self, context_size, embedding_dim, vocab_size, hidden_size, num_layers):
        super(GRULM, self).__init__()
        self.embed_model = load_model(EMBED_MODEL_PATH)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True) #batch_first=True?
        self.linear = nn.Linear(hidden_size*2, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    # Minimize (A*sum(q) + b), a linear combination with the sum of the embedded input
    def forward(self, inputs):
    # Embed the context of a word and sum it into an embedded vector
        embedded_input = self.embedding(inputs)
        # embedded_input = self.embed_model.embed_tensor_input(inputs)

        # Since the network is bidirectional, we need 2 layers
        h0 = Variable(torch.zeros(NUM_LAYERS*2, BATCH_SIZE, HIDDEN_SIZE)).cuda()
        c0 = Variable(torch.zeros(NUM_LAYERS*2, BATCH_SIZE, HIDDEN_SIZE)).cuda()

        # Forward propagate LSTM
        out, _ = self.lstm(embedded_input, (h0, c0))

        # Reshape output
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out#, h

# Simple timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)

        print('The function ran for {0} seconds'.format(time.time() - start))
        return output
    return wrapper

def retrieve_sequences(sequences, vocabulary, sentence):
    split_sentence = sentence.split()
    vocabulary.extend([word for word in sentence.split()])
    for i in range(CONTEXT_SIZE, len(split_sentence) - CONTEXT_SIZE):
        sequence = split_sentence[i-CONTEXT_SIZE:i+CONTEXT_SIZE+1]
        sequences.append(sequence)
    return sequences, vocabulary

# Retrieve context data and the vocabulary of the dialog and captions
def process_data(data):
    sequences = []
    vocabulary = []
    # translator = str.maketrans('', '', string.punctuation)
    # word_cnt = Counter([word for sample in data.values() for sentence in sample['dialog'] for word in sentence[0].split()])
    for sample in data.values():
        dialog = sample['dialog']
        for sentence in dialog:
            sentence_sequences = []
            sentence_sequences, vocabulary = retrieve_sequences(sentence_sequences, vocabulary, sentence[0])
            sequences.append(sentence_sequences)
        caption = sample['caption']
        sentence_sequences = []
        sentence_sequences, vocabulary = retrieve_sequences(sentence_sequences, vocabulary, caption)
        sequences.append(sentence_sequences)

    vocabulary = set(vocabulary)
    vocab_size = len(vocabulary)

    w2i = {word: i for i, word in enumerate(vocabulary)}

    return sequences, vocab_size, w2i

def context_to_index(context, w2i):
    return np.array([w2i[word] for word in context])

def make_data_points(sequences, w2i):
    data_points_input = np.zeros((2*CONTEXT_SIZE+1), dtype=np.long)
    data_points_output = np.zeros((2*CONTEXT_SIZE+1), dtype=np.long)
    for sentence_sequences in sequences:
        for i in range(len(sentence_sequences)-1):
            context_vector = np.array(context_to_index(sentence_sequences[i], w2i))
            target = np.array(context_to_index(sentence_sequences[i+1], w2i))
            data_points_input = np.vstack((data_points_input, context_vector))
            data_points_output = np.vstack((data_points_output, target))

    data_points_input, data_points_output = shuffle(data_points_input, data_points_output)

    return [data_points_input, data_points_output]

@timer
def train_model(data_points, vocab_size, w2i):
    torch.cuda.manual_seed(1)

    model = GRULM(CONTEXT_SIZE, EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, NUM_LAYERS)
    loss_function = nn.CrossEntropyLoss()

    losses = []

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def detach(states):
        return [state.detach() for state in states]

    num_data_points = data_points[0].shape[0]
    num_batches = num_data_points // BATCH_SIZE

    for iteration in range(EPOCHS):

        for i in range(0, num_data_points - BATCH_SIZE, BATCH_SIZE):# - CONTEXT_SIZE*2, CONTEXT_SIZE*2): # Is this really correct? TODO
            if i + BATCH_SIZE >= num_data_points:
                batch_input = data_points[0][i:]
                batch_output = data_points[1][i:]
            else:
                batch_input = data_points[0][i:i+BATCH_SIZE]
                batch_output = data_points[1][i:i+BATCH_SIZE]
            inputs = Variable(torch.from_numpy(batch_input)).cuda()
            targets = Variable(torch.from_numpy(batch_output)).cuda()
            targets = targets.view(-1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()

            if (i) % BATCH_SIZE == 0:
                print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                       (iteration+1, EPOCHS, i/BATCH_SIZE+1, num_batches, loss.data[0], np.exp(loss.data[0])))
    return model

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model

def main():
    _, data_easy, data_hard = rd.read_data()
    sequences, vocab_size, w2i = process_data(data_easy)
    data_points = make_data_points(sequences[:16000], w2i)
    small_set = data_points
    model = train_model(small_set, vocab_size, w2i)
    # save_model(model, MODEL_PATH)
    # model = load_model(model, MODEL_PATH)

    # model = load_model(EMBED_MODEL_PATH)
    # print(model.embed_word_vector(['yachts']))
    # print(model.embed_index_vector([100]))


if __name__ == "__main__":
    main()