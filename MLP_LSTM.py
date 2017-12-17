import read_data as rd
import numpy as np
import LSTM_dialog as ld

import torch
import torch.cuda
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.neural_network import MLPRegressor
from LSTM_dialog import LSTMLM
from itertools import chain

MODEL_PATH_EASY = 'LSTMLM_bidirectional_caption_easy.nn'
MODEL_PATH_HARD = 'LSTMLM_bidirectional_caption_hard.nn'
PREPROCCESS = False
USE_DIALOG = False
BATCH_SIZE = 250
input_size = 2348
hidden_size = 1000
num_classes = 1
num_epochs = 5
learning_rate = 0.002

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.init_weights()

    def init_weights(self):
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, inputs):
        out = self.fc1(inputs)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def retrieve_img_features(img_feature_data, img2id, img_id):
    img_features = np.array(img_feature_data[img2id[str(img_id)]])
    return img_features

# Retrieve context data and the vocabulary of the dialog and captions
def retrieve_data(model, data, img2id, img_feature_data):
    input_data = []
    output_data = []
    for sample in data.values():
        caption = sample['caption']
        split_caption = caption.split()
        context_vector = np.array(ld.context_to_index(split_caption, model.w2i))
        torch_input = Variable(torch.from_numpy(context_vector)).cuda()
        torch_input = torch_input.view(1, -1)
        output, (hn, cn) = model(torch_input)
        hn = hn.view(-1)

        img_ids = sample['img_list']
        img_feature_list = []
        for img_id in img_ids:
            img_feature = np.concatenate([hn.data.cpu().numpy(), retrieve_img_features(img_feature_data, img2id, img_id)], axis=0)
            if img_id == sample['target_img_id']:
                output_data.append(1)
            else:
                output_data.append(0)
            input_data.append(img_feature)
    return np.array(input_data), np.array(output_data)

def save_model(model, path):
    torch.save(model, path)

def main():

    LSTM_model = torch.load(MODEL_PATH_HARD)
    img_feature_data, img2id, train_data_hard, val_data_hard, test_data_hard = rd.read_data('Hard')

    input_data_train, output_data_train = retrieve_data(LSTM_model, train_data_hard, img2id, img_feature_data)
    input_data_test, output_data_test = retrieve_data(LSTM_model, test_data_hard, img2id, img_feature_data)

    print('Data received. Amount of data points in training set: ', len(input_data_train))

    torch.cuda.manual_seed(1)
    mlp = MLP(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    losses = []
    mlp.cuda()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i in range(0, len(input_data_train) - BATCH_SIZE, BATCH_SIZE):# - CONTEXT_SIZE*2, CONTEXT_SIZE*2): # Is this really correct? TODO
            batch_input = input_data_train[i:i+BATCH_SIZE]
            batch_output = output_data_train[i:i+BATCH_SIZE]
            # Convert torch tensor to Variable
            inputs = Variable(torch.from_numpy(batch_input).cuda())
            targets = Variable(torch.from_numpy(batch_output).float().cuda())


            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = mlp(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss)
            loss.backward()
            optimizer.step()

            if (i) % BATCH_SIZE == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i//BATCH_SIZE, len(input_data_train)//BATCH_SIZE, loss.data[0]))

    print('Trained')

    loss_file = open('losses_lstm_torch_hard.txt', 'w')
    for loss in losses:
        loss_file.write("%s\n" % loss.data)
    print("Losses stored")
    save_model(mlp, 'MLP_TORCH_HARD')
    print('Saved')

if __name__ == '__main__':
    main()
