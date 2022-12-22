# PILOT

The Datasets folder contains the Maldonado and Zaho datasets

NLP Techniques
1. FeedForward Neural Network (FFNN)
Folder: tf-idf
a. FFNN-tfidf-binary.py contains the FeedForward Neural Network for binary classification. 
If you want to use with Maldonado Dataset, the line 121 has to be uncommented. For Zaho dataset comments line 121 and uncomment line 122.
b. FFNN-tfidf-multiclass.py contains the FeedForward Neural Network for multi-class classification. 

Note: Before using the related FFNN, please unzip the chosen dataset within binary (or multiclass) folder.

2. Convolutional and Recurrent Neural Networks using word2vec 
Folder: word2vec
a. CNN-word2vec-binary.py (RNN-word2vec-binary.py) contains the Convolutional Neural Network (Recurrent) for binary classification. 
If you want to use with Maldonado Dataset, the line 196 (187 for RNN) has to be uncommented. For Zaho dataset comments line 196 (187 for RNN) and uncomment line 197 (188 for RNN).
b. CNN-word2vec-multiclass.py (RNN-word2vec-multiclass.py) contains the Convolutional Neural Network (Recurrent) for multi-class classification.

Note 1: Before using the CNN or RNN, please download the word2vec-GoogleNews-vectors from the following link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
        and put it to the following folder: word2vec/model 
Note 2: Before using the CNN or RNN, please run the preprocessing-and-split-word2vec.py file to create the Dataset you want to use. See comments 301-303 on how to utilize it.

