import torch
from torch.autograd import Variable
import numpy as np
import read_data as rd
import pickle

from MLP_LSTM import retrieve_img_features
from MLP_LSTM import MLP
import LSTM_dialog as ld
from LSTM_dialog import LSTMLM

import matplotlib.pyplot as plt

EMBED_MODEL_PATH_EASY = 'LSTMLM_bidirectional_caption_easy.nn'
EMBED_MODEL_PATH_HARD = 'LSTMLM_bidirectional_caption_hard.nn'
NEURAL_MODEL_PATH_EASY = 'MLP_TORCH_EASY'
NEURAL_MODEL_PATH_HARD = 'MLP_TORCH_HARD'

def plot_accuracy(top_1_acc, top_5_acc):
    plt.plot(top_1_acc, label="Top 1")
    plt.plot(top_5_acc, label='Top 5')
    plt.show()

def top_accuracy(neural_model, lstm_model, data, img2id, img_feature_data):
    top_1_count = 0
    top_5_count = 0
    top_5_count_list = []
    top_1_count_list = []
    for i, sample in enumerate(data.values()):
        embedded_text = []
        caption = sample['caption']
        split_caption = caption.split()
        context_vector = np.array(ld.context_to_index(split_caption, lstm_model.w2i))
        torch_input = Variable(torch.from_numpy(context_vector)).cuda()
        torch_input = torch_input.view(1, -1)
        output, (hn, cn) = lstm_model(torch_input)
        hn = hn.view(-1)

        img_ids = sample['img_list']
        target_id = sample['target']
        predicted_values = []
        for img_id in img_ids:
            img_feature = np.concatenate([hn.data.cpu().numpy(), retrieve_img_features(img_feature_data, img2id, img_id)], axis=0)

            input_feat = Variable(torch.from_numpy(img_feature).cuda())
            output = neural_model(input_feat)
            predicted_values.extend(output.data)
        predicted_values = np.array(predicted_values)
        top_5_indices = predicted_values.argsort()[-5:][::-1]
        top_1 = top_5_indices[0]
        top_5_ids = []
        if top_1 == target_id:
            top_1_count += 1
        if target_id in top_5_indices:
            top_5_count += 1
        top_1_count_list.append(top_1_count/(i+1))
        top_5_count_list.append(top_5_count/(i+1))
    plot_accuracy(top_1_count_list, top_5_count_list)
    top_1_accuracy = top_1_count/len(data.values())
    top_5_accuracy = top_5_count/len(data.values())
    return top_1_accuracy, top_5_accuracy


def main():
    # lstm_model = torch.load(EMBED_MODEL_PATH_EASY)
    # neural_model = torch.load(NEURAL_MODEL_PATH_EASY)
    # img_feature_data, img2id, train_data_easy, val_data_easy, test_data_easy = rd.read_data('Easy')
    # top_1_accuracy, top_5_accuracy= top_accuracy(neural_model, lstm_model, test_data_easy, img2id, img_feature_data)

    lstm_model = torch.load(EMBED_MODEL_PATH_HARD)
    neural_model = torch.load(NEURAL_MODEL_PATH_HARD)
    img_feature_data, img2id, train_data_hard, val_data_hard, test_data_hard = rd.read_data('Hard')
    top_1_accuracy, top_5_accuracy= top_accuracy(neural_model, lstm_model, test_data_hard, img2id, img_feature_data)

    print(top_1_accuracy)
    print(top_5_accuracy)

if __name__ == '__main__':
    main()
