import torch
import numpy as np
import read_data as rd
import pickle
import bow_dialog as bd
import matplotlib.pyplot as plt


from MLP_cbow import retrieve_img_features
from bow_dialog import CBOW

MODEL_PATH_EASY = 'CBOW_EASY.pth'
MODEL_PATH_HARD = 'CBOW_HARD.pth'
PRE_MODEL_PATH_EASY = 'PREPROCCESS_CBOW_EASY.pth'
PRE_MODEL_PATH_HARD = 'PREPROCCESS_CBOW_HARD.pth'
PREPROCCESS = False

def plot_accuracy(top_1_acc, top_5_acc):
    plt.plot(top_1_acc, label="Top 1")
    plt.plot(top_5_acc, label='Top 5')
    plt.show()

def top_accuracy(neural_model, embed_model, data, img2id, img_feature_data):
    top_1_count = 0
    top_5_count = 0
    top_5_count_list = []
    top_1_count_list = []
    for i, sample in enumerate(data.values()):
        embedded_text = []
        dialog = sample['dialog']
        for sentence in dialog:
            if PREPROCCESS:
                sentence = bd.get_filtered_sentence(sentence[0])
            else:
                sentence = sentence[0]
            embedded_text.extend(sentence.split())
        caption = sample['caption']
        if PREPROCCESS:
            caption = bd.get_filtered_sentence(caption)
        embedded_text.extend(caption.split())
        embedded_text = embed_model.embed_word_vector(embedded_text)

        img_ids = sample['img_list']
        target_id = sample['target']
        predicted_values = []
        for img_id in img_ids:
            img_feature = np.concatenate([embedded_text, retrieve_img_features(img_feature_data, img2id, img_id)], axis=0)
            predicted_values = np.concatenate([predicted_values, neural_model.predict([img_feature])])
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
    embed_model = torch.load(MODEL_PATH_EASY)
    img_feature_data, img2id, train_data_easy, val_data_easy, test_data_easy = rd.read_data('Easy')
    img_feature_data, img2id, train_data_hard, val_data_hard, test_data_hard = rd.read_data('Hard')

    MLPR = pickle.load(open('MLPR.p', 'rb'))
    # MLPR = pickle.load(open('PREPROCCESSED_EASY_MLPR_2.p', 'rb'))
    # MLPR = pickle.load(open('HARD_MLPR.p', 'rb'))
    # MLPR = pickle.load(open('EASY_MLPR.p', 'rb'))

    top_1_accuracy, top_5_accuracy= top_accuracy(MLPR, embed_model, test_data_easy, img2id, img_feature_data)
    print(top_1_accuracy)
    print(top_5_accuracy)


if __name__ == '__main__':
    main()