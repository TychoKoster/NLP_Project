from collections import Counter
from sklearn.utils import shuffle

import torch.cuda
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from nltk.corpus import stopwords
#from stemming.porter2 import stem

import read_data as rd
import numpy as np
import sklearn
import pickle
import string
import time

NUM_LAYERS = 1
SEQ_LEN = 3
EMBEDDING_DIM = 128
EPOCHS = 5
MODEL_PATH = 'LSTMLM_bidirectional_caption_easy.nn'
BATCH_SIZE = 1
HIDDEN_SIZE = 150
LEARNING_RATE = 0.01
EASY = 'Easy'
HARD = 'Hard'
BIDRECTIONAL = True
USE_DIALOG = False
#SENTENCES_USED = 55000 # There are 5000 sentences in the captions and 50000 in the dialog for the easy set

class LSTMLM(nn.Module):
    def __init__(self, seq_len, embedding_dim, vocab_size, hidden_size, num_layers, bidirectional, w2i):
        super(LSTMLM, self).__init__()
        self.w2i = w2i

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=bidirectional) #batch_first=True?
        if(bidirectional):
            self.linear = nn.Linear(hidden_size*2, vocab_size)
        else:
            self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
    # Embed the context of a word and sum it into an embedded vector
        embedded_input = self.embedding(inputs)

        # If the network is bidirectional, we need 2 layers
        if(BIDRECTIONAL):
            h0 = Variable(torch.zeros(NUM_LAYERS*2, BATCH_SIZE, HIDDEN_SIZE)).cuda()
            c0 = Variable(torch.zeros(NUM_LAYERS*2, BATCH_SIZE, HIDDEN_SIZE)).cuda()
        else:
            h0 = Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)).cuda()
            c0 = Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)).cuda()

        # Forward propagate LSTM
        out, h = self.lstm(embedded_input, (h0, c0))

        # Reshape output
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out, h

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

def retrieve_sequences(sequences, vocabulary, sentence):
    split_sentence = sentence.split()
    vocabulary.extend([word for word in sentence.split()])
    for i in range(0, len(split_sentence) - SEQ_LEN):
        sequence = split_sentence[i:i+SEQ_LEN]
        sequences.append(sequence)
    return sequences, vocabulary

# Retrieve context data and the vocabulary of the dialog and captions
def process_data(data):
    sequences = []
    vocabulary = []
    # translator = str.maketrans('', '', string.punctuation)
    # word_cnt = Counter([word for sample in data.values() for sentence in sample['dialog'] for word in sentence[0].split()])
    for sample in data.values():
        if(USE_DIALOG):
            dialog = sample['dialog']
            for sentence in dialog:
                sentence_sequences = []
                sentence_sequences, vocabulary = retrieve_sequences(sentence_sequences, vocabulary, sentence[0])
                sequences.append(sentence_sequences)
        caption = sample['caption']
        sentence_sequences = []
        sentence_sequences, vocabulary = retrieve_sequences(sentence_sequences, vocabulary, caption)
        sequences.append(sentence_sequences)

    vocabulary.extend(['UNKNOWN'])
    vocabulary = set(vocabulary)
    vocab_size = len(vocabulary)

    w2i = {word: i for i, word in enumerate(vocabulary)}

    return sequences, vocab_size, w2i

def context_to_index(context, w2i):
    index_vec = []
    for word in context:
        if word in w2i.keys():
            index_vec.append(w2i[word])
        else:
            index_vec.append(w2i['UNKNOWN'])
    return index_vec

def make_data_points(sequences, w2i):
    data_points_input = np.zeros((SEQ_LEN), dtype=np.long)
    data_points_output = np.zeros((SEQ_LEN), dtype=np.long)
    for sentence_sequences in sequences:
        contexts = []
        targets = []
        for i in range(len(sentence_sequences)-1):
            context_vector = np.array(context_to_index(sentence_sequences[i], w2i))
            contexts.append(context_vector)
            target = np.array(context_to_index(sentence_sequences[i+1], w2i))
            targets.append(target)
        data_points_input = np.vstack((data_points_input, contexts))
        data_points_output = np.vstack((data_points_output, targets))

    data_points_input, data_points_output = shuffle(data_points_input, data_points_output)

    return [data_points_input, data_points_output]

@timer
def train_model(data_points, vocab_size, w2i):
    torch.cuda.manual_seed(1)

    model = LSTMLM(SEQ_LEN, EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, NUM_LAYERS, BIDRECTIONAL, w2i)
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
            batch_input = data_points[0][i:i+BATCH_SIZE]
            batch_output = data_points[1][i:i+BATCH_SIZE]
            inputs = Variable(torch.from_numpy(batch_input)).cuda()
            targets = Variable(torch.from_numpy(batch_output)).cuda()
            targets = targets.view(-1)

            optimizer.zero_grad()
            outputs, h = model(inputs)
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
    _, _, train_data, val_data, test_data = rd.read_data(EASY)
    all_data = dict(train_data)
    all_data.update(val_data)
    all_data.update(test_data)
    sequences, _, _ = process_data(train_data)
    _, vocab_size, w2i = process_data(all_data)
    #data_points = make_data_points(sequences[:SENTENCES_USED], w2i)
    data_points = make_data_points(sequences, w2i)
    model = train_model(data_points, vocab_size, w2i)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
