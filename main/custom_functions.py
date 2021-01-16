# Custom functions for data prepraration and inference generation

import pandas as pd
import numpy as np
import re
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder


# Reading dataframe
def read_data(path):
  data = pd.read_csv(path)
  data = data.fillna('')
  return data


# Extract all stopwords from a text file and store them in a list.
def get_stopwords(textfile):
  stopwords = []
  file = open(textfile, "r")
  for line in file:
    line = line.replace("\n","")
    stopwords.append(line)
  file.close()
  return stopwords


# Clean hashtags, mentions, special/unwnated/unprintable characters from the tweet.  
def clean_text(text):
    
    # Eliminate unprintable characters
    text = ''.join(x for x in text if x.isprintable())

    # Eliminate urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Eliminate html elements
    text = re.sub(r'<.*?>', '', text)

    # Eliminate punctuations
    text = re.sub('[!#?,.:";-@#$%^&*_~<>()-]', '', text)
  
    # Eliminating hashtags (keeping original word)
    text = re.sub(r'#', '', text) 
      
    # Eliminating special characters like retweet text "RT" 

    text = re.sub(r'(\\x(.)*)', '',text)
    text = re.sub(r'\\n|\\t|\\n\\n', ' ', text)
    text = re.sub(r"b'RT|b'|b RT|b\"RT|b", "", text)
    text = re.sub("[@#$%^&*)(}{|/><=+=_:\"\\\\]+"," ",text).strip()

    return text


# Tokenize input text to integer-id sequences using keras Tokenizer.
def tokenize_text(corpus, x_train, x_val):
  tokenizer = Tokenizer(oov_token = "[OOV]")
  tokenizer.fit_on_texts(corpus)
  x_train_tokenized = tokenizer.texts_to_sequences(x_train)
  x_val_tokenized = tokenizer.texts_to_sequences(x_val)

  return tokenizer, x_train_tokenized, x_val_tokenized


# Padding all text sequences to get uniform-length input. 
def pad_text_sequence(x_tokenized, pad_length, pad_type = "post", truncate_type = "post"):

  x_padded = sequence.pad_sequences(sequences = x_tokenized, 
                                          padding = pad_type,
                                          truncating = truncate_type,
                                          maxlen = pad_length)  
  return x_padded


# Enconde target variabels (class labels) to integers.
def get_label_encoding(labels):
  le = LabelEncoder()
  le.fit(np.unique(labels))
  label_encodings = le.transform(labels)
  
  print("Mapping:")
  for c,l in zip(le.classes_, label_encodings):
    print(c,":",l,"\t")
  print("\n")

  label_encodings = label_encodings.reshape(label_encodings.shape[0], -1)
  return label_encodings


# Extracts words and their corresponding vectors from a .vec/.txt file.
# Returns a word x embedding matrix 
def get_word_embeddings(filepath, vocab, embedding_dimension):

  word_embeddings = np.zeros((len(vocab) + 1, embedding_dimension))
#   word_embeddings = []
  embedding_file = open(filepath, "r", encoding = "utf8")
  count = 0

  for line in embedding_file:
    line = line.split()
    word = line[0]
    if word in vocab:
      word_vector = np.asarray(line[1:], dtype = "float32")
      if len(word_vector) == embedding_dimension:
        word_embeddings[vocab[word]] = word_vector
      else:
        print('\nVector size does not match with embedding_dimension:\t', word_vector)
      count+=1

  print("Total word embeddings read: {}\n".format(count))
  embedding_file.close()
  return word_embeddings     


# Different methods of getting sentence/document embeddings from word vectors. 
def get_sentence_embedding(embedding_matrix, corpus, option='bow'):
    all_sentence_embeddings = []
    if option == 'bow':
        for row in corpus:
            sentence_embedding = np.zeros(300)
            for loc, value in list(zip(row.indices, row.data)):
                sentence_embedding = sentence_embedding + value*embedding_matrix[loc]
            if row.data.shape[0] != 0:
                sentence_embedding = sentence_embedding/row.data.shape[0]
            all_sentence_embeddings.append(sentence_embedding)
        all_sentence_embeddings = np.array([np.array(x) for x in all_sentence_embeddings])
        return all_sentence_embeddings
        
    elif option == 'tfidf':
        for row in corpus:
            sentence_embedding = np.zeros(300)
            for loc, value in list(zip(row.indices, row.data)):
                sentence_embedding = sentence_embedding + value*embedding_matrix[loc]
            all_sentence_embeddings.append(sentence_embedding)
        all_sentence_embeddings = np.array([np.array(x) for x in all_sentence_embeddings])
        return all_sentence_embeddings
    
    else:
        print("Invalid option")
        return text


# Evaluate model performace timeline.
def plot_results(model_history):

  # Train vs Val accuracy
  plt.title("model accuracy timeline")
  plt.xlabel("epoch")
  plt.ylabel("acc")
  plt.plot(model_history.history['acc'])
  plt.plot(model_history.history['val_acc'])
  plt.legend(["train", "val"])
  plt.show()

  # Train vs Val loss
  plt.title("model loss timeline")
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.plot(model_history.history['loss'])
  plt.plot(model_history.history['val_loss'])
  plt.legend(["train", "val"])
  plt.show()


# Save model and model weights to a local file.
def save_model(model, name):
  try:
    model.save(name + ".h5")
    model.save_weights(name + "_weights.h5")
    print("Saved as ",name)
  except Exception as e:
    print(e)
