# added GPU support because I thought training was slow, but GPU seems even slower?
# Shouldn't we use cross entropy as loss?
# Shuffle data?
# Still errors...

from collections import Counter

import torch
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
CONTEXT_SIZE = 1
EMBEDDING_DIM = 64
EPOCHS = 10
MODEL_PATH = 'GRULM.pt'
BATCH_SIZE = 64
HIDDEN_SIZE = 128
LEARNING_RATE = 0.002

class GRULM(nn.Module):
    def __init__(self, context_size, embedding_dim, vocab_size, hidden_size, num_layers):
        super(GRULM, self).__init__()
        #self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #self.linear = nn.Linear(embedding_dim, vocab_size)

        #self.gru = nn.GRU(EMBEDDING_FEAT*CONTEXT_SIZE*2, hidden_size)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True) #batch_first=True?
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    # Minimize (A*sum(q) + b), a linear combination with the sum of the embedded input
    def forward(self, inputs, hidden):
    # Embed the context of a word and sum it into an embedded vector
        embedded_input = self.embedding(inputs)

        # Forward propagate GRU
        out, h = self.gru(embedded_input, hidden)

        # Reshape output
        out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))
        #rearranged = hn.view(hn.size()[1], hn.size(2))

        out = self.linear(out)
        return out
        #print(embedded_input)
        #embedded = nn.GRU()
        #GRU model here

    #embedded = embedded_input.sum(0)
    # Reshape to prevent input error when measuring loss
    #embedded = embedded.view(1,-1)
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
    #context_vector = torch.cuda.LongTensor(indices)
    return indices

@timer
def train_model(context_data, vocab_size, w2i):
    torch.cuda.manual_seed(1)
    losses = []
    # CBOW minimizes the negative log likelihood
    loss_function = nn.CrossEntropyLoss()
    model = GRULM(CONTEXT_SIZE, EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, NUM_LAYERS)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def detach(states):
        return [state.detach() for state in states]

    print(len(context_data))

    # Should only run this part once, and already be in correct format
    context_indices = np.zeros((CONTEXT_SIZE*2, len(context_data))) # this correct?
    targets = np.zeros((1, len(context_data)))
    count = 0
    for context, target in context_data:
        context_indices[:, count] = context_to_index(context, w2i)
        targets[:, count] = [w2i[target]]
        count += 1

    print("done")

    for iteration in range(EPOCHS):
        states = (Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)),
              Variable(torch.zeros(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)))
        total_loss = torch.cuda.FloatTensor([0])
    #count = 0
        booltje = False
        #for context, target in context_data:#
        for i in range(0, context_indices.shape[1] - BATCH_SIZE, BATCH_SIZE):
            #context_batch = np.zeros((BATCH_SIZE, 1))
    #context_batch = context_indices[:, t:t+b](
            #context_batch = Variable(context_batch)
            inputs = torch.from_numpy(context_indices[:, i:i+BATCH_SIZE]).cuda()
            inputs.type(torch.cuda.LongTensor)
            inputs = Variable(inputs)
            targets = torch.from_numpy(targets[:, i:i+BATCH_SIZE]).cuda()
            targets.type(torch.cuda.LongTensor)
            targets = Variable(targets)
            # if(not booltje):
            #     print("c", context_vector)
            #     print("s", context_vector.data.shape)
            #     print("t", target)
            #     print("s", target.data.shape)
            #     booltje = True

            states = detach(states)
            outputs, states = model(inputs, states)
            #word_embedding_prediction = model(inputs, Variable(torch.randn(NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE))) # I'm not sure what this output is meant to be?
            #word_embedding_prediction = model(inputs)
            #loss = loss_function(probabilities, targets) # Not sure what the loss function is
            loss = loss_function(probabilities, targets.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.data

            step = (i+1) // BATCH_SIZE
            if step % 100 == 0:
                print ('Epoch [%d/%d], Step[%d/%d], Loss: %.3f, Perplexity: %5.2f' %
                       (iteration+1, EPOCHS, step, context_indices.shape[1] // BATCH_SIZE.exp(loss.data[0])))

        #    t += BATCH_SIZE
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
    small_set = context_data[:1000]
    model = train_model(small_set, vocab_size, w2i)
    #save_model(model, MODEL_PATH)
    # model = load_model(MODEL_PATH)

if __name__ == "__main__":
    main()
