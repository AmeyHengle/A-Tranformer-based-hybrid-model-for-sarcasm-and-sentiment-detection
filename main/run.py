import pandas as pd
import numpy as np
from time import process_time
import pickle
from tqdm import tqdm
import sys
import configparser

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# from transformer_model import myBERT
from custom_models import *
from transformer_models import *
from custom_functions import *

config = configparser.ConfigParser()
config.read("config.txt")

text_label = config['labels']['text_label']
class_label = config['labels']['class_label']
max_seq_len = int(config["variables"]["max_seq_len"])
max_seq_len_arabert = int(config["arabert_params"]["max_seq_len_arabert"])


if __name__ == "__main__":
    
    D_train = read_data(config["data"]["train"])
    D_val = read_data(config["data"]["val"])
    x_train = D_train[text_label].values.tolist()
    x_val = D_val[text_label].values.tolist()    
    tokenizer, x_train_padded, x_val_padded = prepare_training_data(x_train, x_val, max_seq_len = int(config["variables"]["max_seq_len"])
                                                                    , padding_type = config["variables"]["padding_type"],
                                                                    truncating_type = config["variables"]["truncating_type"])

    y_train, y_val = D_train[class_label].values.tolist(), D_val[class_label].values.tolist()
    y_train = get_label_encoding(y_train)
    y_val = get_label_encoding(y_val)
    num_labels = len(np.unique(y_train))
    print("\nOutput class shape: ",y_train.shape, y_val.shape)
    print("\nTotal classes: ", num_labels)


    # Load word embeddings from file
    vocab = tokenizer.word_index
    file_name = config["pretrained_embeddings"]["embedding_file_path"]
    embedding_dimension = int(config["variables"]["embedding_dimension"])

    embedding_dict_file = file_name
    embedding_matrix = get_embedding_matrix(vocab, embedding_dict_file = embedding_dict_file, embedding_dimension = embedding_dimension)

    # Validate embedding_matrix shape
    print("\nTotal Vocab:",len(vocab), "\nEmbeddings:",embedding_matrix.shape[0] -1)


    # Load AraBERT model
    arabert = myBERT(model_name = config["arabert_params"]["model_name"],
                tokenizer_name = config["arabert_params"]["tokenizer_name"],
                max_seq_len = max_seq_len_arabert)


    # Getting input features (input_ids and attention_masks) for the AraBERT model
    x_train_features = arabert.get_input_features(x_train, preprocess=True)

    x_val_features = arabert.get_input_features(x_val, preprocess=True)

    
    text_label = config['labels']['text_label']
    class_label = config['labels']['class_label']
    
    D_train = read_data(config["data"]["train"])
    D_val = read_data(config["data"]["val"])
    
    x_train = D_train[text_label].values.tolist()
    x_val = D_val[text_label].values.tolist()    
    tokenizer, x_train_padded, x_val_padded = prepare_training_data(x_train, x_val, max_seq_len = int(config["variables"]["max_seq_len"])
                                                                    , padding_type = config["variables"]["padding_type"],
                                                                    truncating_type = config["variables"]["truncating_type"])
    
    y_train, y_val = D_train[class_label].values.tolist(), D_val[class_label].values.tolist()
    y_train = get_label_encoding(y_train)
    y_val = get_label_encoding(y_val)
    num_labels = len(np.unique(y_train))
    print("\nOutput class shape: ",y_train.shape, y_val.shape)
    print("\nTotal classes: ", num_labels)
    
    
    # Load word embeddings from file
    vocab = tokenizer.word_index
    file_name = config["pretrained_embeddings"]["embedding_file_path"]
    embedding_dimension = int(config["variables"]["embedding_dimension"])

    embedding_dict_file = file_name
    embedding_matrix_mj = get_embedding_matrix(vocab, embedding_dict_file = embedding_dict_file, embedding_dimension = embedding_dimension)
    
    # Validate embedding_matrix shape
    print("\nTotal Vocab:",len(vocab), "\nEmbeddings:",embedding_matrix_mj.shape[0] -1)
    
    
    # Load AraBERT model
    arabert = myBERT(model_name = config["arabert_params"]["model_name"],
                 tokenizer_name = config["arabert_params"]["tokenizer_name"],
                 max_seq_len = int(config["arabert_params"]["max_seq_len_arabert"]))

    
    # Getting input features (input_ids and attention_masks) for the AraBERT model
    x_train_features = arabert.get_input_features(x_train, preprocess=True)
    x_val_features = arabert.get_input_features(x_val, preprocess=True)
    

    aux_model = auxillaryModel(embedding_matrix = embedding_matrix, 
                            max_seq_len = max_seq_len,
                            num_classes= num_labels)

    model = aux_model.load_model(model_name = config["model_params"]["auxillary_model_name"])



    if bool(config['model_params']['ensemble']) == True:
    print("Training ensemble model")

    # Input features to the BERT model 
    input_ids_in = tf.keras.layers.Input(shape=(max_seq_len_arabert,), name='input_token', dtype='int32')
    attention_masks_ids_in = tf.keras.layers.Input(shape=(max_seq_len_arabert,), name='m"asked_token', dtype='int32')

    # Load model
    transformer_model = arabert.load_model(num_labels=num_labels)
    embedding_layer_ab = transformer_model(input_ids_in, attention_masks_ids_in)[0]

    # Input 1
    cls_token = embedding_layer_ab[:, 0, :]        # Use just the CLS token for classification
    X = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer_ab)
    model1 = tf.keras.Model(inputs=[input_ids_in, attention_masks_ids_in], outputs = cls_token)

    # Input 2
    aux_model = auxillaryModel(embedding_matrix = embedding_matrix, 
                                max_seq_len = max_seq_len,
                                num_classes= num_labels)
    model2 = aux_model.load_model(model_name = config["model_params"]["auxillary_model_name"])


    # Concat
    concat = tf.keras.layers.concatenate([model1.output, model2.output])

    # Output Layer
    output = tf.keras.layers.Dense(num_labels, activation = "softmax")(concat)

    # Build model
    model = tf.keras.Model(inputs = [model1.input, model2.input], outputs = output)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model_history = model.fit([[x_train_features[0], x_train_features[1]], x_train_padded], np.array(y_train),
                            validation_data = ([[x_val_features[0], x_val_features[1]], x_val_padded], np.array(y_val)),
                            epochs = 1, batch_size = 128)

    else:
    print("Training pretrained-AraBERT model")

    # Input features to the BERT model 
    input_ids_in = tf.keras.layers.Input(shape=(max_seq_len_arabert,), name='input_token', dtype='int32')
    attention_masks_ids_in = tf.keras.layers.Input(shape=(max_seq_len_arabert,), 
                                                    name='m"asked_token', dtype='int32')
    # Load model
    transformer_model = arabert.load_model(num_labels=num_labels, sequence_classifier=True)
    X = transformer_model(input_ids_in, attention_masks_ids_in)
    model = tf.keras.Model(inputs = [input_ids_in, attention_masks_ids_in], outputs = X)

    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model_history = model.fit([x_train_features[0], x_train_features[1]], np.array(y_train),
                                validation_data = ([x_val_features[0], x_val_features[1]], np.array(y_val)),
                                epochs = 3, batch_size = 64)