import read_data as rd
import pickle
import torch 
import numpy as np
import bow_dialog as bd

from sklearn.neural_network import MLPRegressor
from bow_dialog import CBOW
from itertools import chain

MODEL_PATH_EASY = 'CBOW_EASY.pth'
MODEL_PATH_HARD = 'CBOW_HARD.pth'
PREPROCCESS = True

def retrieve_img_features(img_feature_data, img2id, img_id):
    img_features = np.array(img_feature_data[img2id[str(img_id)]])
    return img_features

# Retrieve context data and the vocabulary of the dialog and captions
def retrieve_data(model, data, img2id, img_feature_data):
    input_data = []
    output_data = []
    for sample in data.values():
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
        embedded_text = model.embed_word_vector(embedded_text)
        img_ids = sample['img_list']
        img_feature_list = []
        for img_id in img_ids:
            img_feature = np.concatenate([embedded_text, retrieve_img_features(img_feature_data, img2id, img_id)], axis=0)
            if img_id == sample['target_img_id']:
                output_data.append(1)
            else:
                output_data.append(0)
            input_data.append(img_feature)
    return np.array(input_data), np.array(output_data)

# Plot with x axis variable input data, y axis top-1 or top-5 accuracy
def main():
    embed_model = torch.load(MODEL_PATH_EASY)
    
    img_feature_data, img2id, train_data_easy, val_data_easy, test_data_easy = rd.read_data('Easy')
    _, _, train_data_hard, val_data_hard, test_data_hard = rd.read_data('Hard')
    # data = list(train_data_easy.values())
    # data.extend(list(val_data_easy.values()))
    # data.extend(list(test_data_easy.values()))
    # _, _, w2i = bd.process_data(data)

    input_data_train, output_data_train = retrieve_data(embed_model, train_data_easy, img2id, img_feature_data)
    input_data_test, output_data_test = retrieve_data(embed_model, test_data_easy, img2id, img_feature_data)
    print('Data received')

    # for i in range(100, 1500, 100):
    MLPR = MLPRegressor(hidden_layer_sizes=(1000,), activation='logistic', solver='adam', alpha=0.0002, max_iter=10, learning_rate_init=0.0005)
    MLPR.fit(input_data_train, output_data_train)
    print('Trained')
    score = MLPR.score(input_data_test, output_data_test)
    print(score)
    # with open('scores.txt', 'a') as f:
    #     line = '{0}, {1} \n'.format(i, score)
    #     f.write(line)
    if PREPROCCESS:
        pickle.dump(MLPR, open('PREPROCCESSED_MLPR.p', 'wb'))
    else:
        pickle.dump(MLPR, open('MLPR.p', 'wb'))
    print('Saved')

    # models.append((MLPR, MLPR.score(X_train, y_train) + MLPR.score(X_test, y_test)))
    # models.sort(key=lambda x:x[1], reverse=True)

    # pickle.dump(models[0][0], open('torcs-server/torcs-client/MLPR_no_opponents.p', 'wb+'))

if __name__ == '__main__':
    main()