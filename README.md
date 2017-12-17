# NLP_Project
## Description
This project is part of the Natural Language Processing 1 course of the MSc Ai. In this project, two straightforward models for the purpose of dialogue-based image retrieval model are described. Text and image features are combined and used to predict the correct image based on given dialogue. The text features are embedded using two different models, a Continuous Bag of Words model and a Long short-term memory network. In order to compare the performance of the two models, top-1 and top-5 accuracy were used. It is shown that a CBOW model with preprocessing of the text features performs significantly better than without the preprocessing. An LSTM model might outperform the CBOW model, but additional testing is required.

## Files
***CBOW_training.py*** Trains the CBOW model with a predefined dataset.

***LSTM_dialog.py*** Trains the LSTM model with a predefined dataset.

***Accuracy_model.py*** Plots the top-1 and top-5 accuracy of the CBOW combined with MLPR model.

***Accuracy_mode_lstml.py*** Plots the top-1 and top-5 accuracy of the LSTM combined with MLPR model.

***MLP_cbow.py*** Trains the MLPR model with CBOW embedding on the text features. 

***MLP_LSTM.py*** Trains the MLPR model with LSTM embedding on the text features. 

