import read_data as rd
import pickle
import torch 
import numpy as np

from sklearn.neural_network import MLPRegressor
from bow_dialog import CBOW

MODEL_PATH_EASY = 'CBOW_EASY.pth'
MODEL_PATH_HARD = 'CBOW_HARD.pth'

def retrieve_img_features(img_feature_data, img2id, img_id):
    img_features = np.array(img_feature_data[img2id[str(img_id)]])
    return np.reshape(img_features, (1, len(img_features)))

# Retrieve context data and the vocabulary of the dialog and captions
def retrieve_data(model, data, img2id, img_feature_data):
    input_data = []
    output_data = []
    for i, sample in enumerate(data.values()):
        input_data.append([])
        output_data.append(sample['target_img_id'])
        dialog = sample['dialog']
        for sentence in dialog:
            input_data[i].extend(sentence[0].split())
        input_data[i].extend(sample['caption'].split())
        input_data[i] = model.embed_word_vector(input_data[i])
        img_ids = sample['img_list']
        for j, img_id in enumerate(img_ids):
            if j == 0:
                img_feature_list = retrieve_img_features(img_feature_data, img2id, img_id)
            else:
                img_feature_list = np.concatenate([img_feature_list, retrieve_img_features(img_feature_data, img2id, img_id)], axis=1)
        input_data[i] = np.concatenate([img_feature_list, input_data[i]], axis=1)
    return np.array(input_data), np.array(output_data)

# Plot with x axis variable input data, y axis top-1 or top-5 accuracy
def main():
    embed_model = torch.load(MODEL_PATH_EASY)
  
    img_feature_data, img2id, train_data_easy, val_data_easy, test_data_easy = rd.read_data('Easy')
    _, _, train_data_hard, val_data_hard, test_data_hard = rd.read_data('Hard')

    input_data_train, output_data_train = retrieve_data(embed_model, train_data_easy, img2id, img_feature_data)

    MLPR = MLPRegressor(hidden_layer_sizes=(10), activation='logistic', solver='lbfgs', alpha=0.0005, max_iter=1)
    print(input_data_train.reshape(input_data_train))
    # MLPR.fit(input_data_train, output_data_train)

    # print(MLPR.predict(input_data_train[0]))
    # print(MLPR.score(X_test, y_test))
    # models.append((MLPR, MLPR.score(X_train, y_train) + MLPR.score(X_test, y_test)))
    # models.sort(key=lambda x:x[1], reverse=True)

    # pickle.dump(models[0][0], open('torcs-server/torcs-client/MLPR_no_opponents.p', 'wb+'))

if __name__ == '__main__':
    main()