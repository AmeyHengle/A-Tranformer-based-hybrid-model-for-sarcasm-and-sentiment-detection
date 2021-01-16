# Custom functions for data prepraration and inference generation

import pandas as pd
import numpy as np
import re
import string
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
def clean_text(text, stop_words): 
  START_OF_LINE = r"^"
  OPTIONAL = "?"
  ANYTHING = "."
  ZERO_OR_MORE = "*"
  ONE_OR_MORE = "+"

  SPACE = "\s"
  SPACES = SPACE + ONE_OR_MORE
  NOT_SPACE = "[^\s]" + ONE_OR_MORE
  EVERYTHING_OR_NOTHING = ANYTHING + ZERO_OR_MORE

  ERASE = ""
  FORWARD_SLASH = "\/"
  NEWLINES = r"[\r\n]"

  arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)

  punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
  print(str(text))
  #mentions
  text = re.sub('@[\w]+','', text)

  #RT
  RE_TWEET = "RT" + SPACES
  text = re.sub(RE_TWEET,'', text)

  #hyperlink
  HYPERLINKS = ("http" + "s" + OPTIONAL + ":" + FORWARD_SLASH + FORWARD_SLASH
            + NOT_SPACE + NEWLINES + ZERO_OR_MORE)
  text = re.sub(HYPERLINKS,ERASE, text)

  #hash
  text = re.sub('#',ERASE, text)

  #punctuation
  translator = str.maketrans('', '', punctuations)
  text = text.translate(translator)

  #diacritics
  text = re.sub(arabic_diacritics, '', text)

  #longation
  text = re.sub("[إأآا]", "ا", text)
  text = re.sub("ى", "ي", text)
  text = re.sub("ؤ", "ء", text)
  text = re.sub("ئ", "ء", text)
  text = re.sub("ة", "ه", text)
  text = re.sub("گ", "ك", text)

  #stopwords
  text = ' '.join(word for word in text.split() if word not in stop_words)

  #emojis and emoticons
  #-----
  

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
  le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
  print(le_name_mapping)
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

    
# Save Fasttext trained embeddings to a .vec file.
def save_fasttext_embeddings(model, output_file):
    file = open(output_file, "w",encoding='utf')
    words = model.get_words()
    print('Input Vocab:\t',str(len(words)), "\nModel Dimensions: ",str(model.get_dimension()))
    cnt = 0
    for w in words:
        v = model.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            row = w + vstr + "\n"
            file.write(row)
            cnt = cnt + 1
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass
    print('Total words processed: ',cnt)
    

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
