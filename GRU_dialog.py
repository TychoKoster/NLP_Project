# added GPU support because I thought training was slow, but GPU seems even slower?
# Shouldn't we use cross entropy as loss?
# Shuffle data?

from collections import Counter

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
EMBEDDING_DIM = 128
EPOCHS = 5
MODEL_PATH = 'GRULM.pt'
BATCH_SIZE = 256
HIDDEN_SIZE = 1024
LEARNING_RATE = 0.002

class GRULM(nn.Module):
    def __init__(self, context_size, embedding_dim, vocab_size, hidden_size, num_layers):
        super(GRULM, self).__init__()
        #self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #self.linear = nn.Linear(embedding_dim, vocab_size)

        #self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True) #batch_first=True?
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True) #batch_first=True?
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        #hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          #autograd.Variable(torch.randn((1, 1, 3))))

    # Minimize (A*sum(q) + b), a linear combination with the sum of the embedded input
    def forward(self, inputs, hidden):
    # Embed the context of a word and sum it into an embedded vector
        #print("input", inputs)
        embedded_input = self.embedding(inputs)
        #print("e", embedded_input)
        #embedded_input = embedded_input.view(BATCH_SIZE, CONTEXT_SIZE, -1)

        # Forward propagate GRU
        out, h = self.lstm(embedded_input, hidden)

        # Reshape output
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        #out = out.contiguous().view(out.size()[1], out.size(2))
        out = self.linear(out)
        return out, h
        #print(embedded_input)
        #embedded = nn.GRU()
        #GRU model here

    # Return the log probabilities
    #return F.log_softmax(self.linear(embedded))

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
    data_points = []
    data_points_input = np.zeros((2*CONTEXT_SIZE+1), dtype=np.long)
    data_points_output = np.zeros((2*CONTEXT_SIZE+1), dtype=np.long)
    for sentence_sequences in sequences:
        for i in range(len(sentence_sequences)-1):
            data_point = []
            context_vector = np.array(context_to_index(sentence_sequences[i], w2i))
            target = np.array(context_to_index(sentence_sequences[i+1], w2i))
            data_points_input = np.vstack((data_points_input, context_vector))
            data_points_output = np.vstack((data_points_output, target))
    data_points = [data_points_input, data_points_output]

    return data_points

@timer
def train_model(data_points, vocab_size, w2i):
    torch.cuda.manual_seed(1)
    #torch.manual_seed(1)
    # CBOW minimizes the negative log likelihood

    model = GRULM(CONTEXT_SIZE, EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, NUM_LAYERS)
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.NLLLoss()

    losses = []

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def detach(states):
        return [state.detach() for state in states]

    #print(data_points[0].shape)
    num_data_points = data_points[0].shape[0]
    #print(len(data_points))
    #print("n", num_data_points)
    num_batches = num_data_points // BATCH_SIZE

    #print("hi")

    for iteration in range(EPOCHS):
        #total_loss = torch.Tensor([0])

        #print(iteration)

        states = (Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)).cuda(),
              Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE).cuda()))

        for i in range(0, num_data_points - BATCH_SIZE, BATCH_SIZE):# - CONTEXT_SIZE*2, CONTEXT_SIZE*2): # Is this really correct? TODO
            batch_input = data_points[0][i:i+BATCH_SIZE]
            batch_output = data_points[1][i:i+BATCH_SIZE]

            inputs = Variable(torch.from_numpy(batch_input)).cuda()
            #inputs = inputs.view(-1)
            targets = Variable(torch.from_numpy(batch_output)).cuda()
            targets = targets.view(-1)

            states = detach(states)
            outputs, states = model(inputs, states)
            loss = loss_function(outputs, targets)


            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

            optimizer.step()
            #total_loss += loss.data

            step = (i+1) // BATCH_SIZE
            #print(step)
            if step % BATCH_SIZE == 0:
                print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                       (iteration+1, EPOCHS, step, num_batches, loss.data[0], np.exp(loss.data[0])))

        #losses.append(total_loss)11
        #print("Total loss epoch {0}: {1}".format(iteration, total_loss))

    return model

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model

def main():
    _, data_easy, data_hard = rd.read_data()
    sequences, vocab_size, w2i = process_data(data_easy)
    data_points = make_data_points(sequences[:8192], w2i)
    #print(data_points[0:10])
    small_set = data_points
    model = train_model(small_set, vocab_size, w2i)
    #save_model(model, MODEL_PATH)
    # model = load_model(MODEL_PATH)

if __name__ == "__main__":
    main()
