import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Embedding, Dense,SpatialDropout1D, GlobalMaxPooling1D,GlobalAveragePooling2D, MaxPooling1D, Conv1D, Dropout, GlobalAveragePooling1D,LSTM, Bidirectional, TimeDistributed, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

from transformers import TFDistilBertModel
from transformers import TFAutoModel
from transformers import TFDistilBertForSequenceClassification, DistilBertConfig
from transformers import AutoConfig, TFAutoModelForSequenceClassification
from transformers import DistilBertTokenizer, RobertaTokenizer, AutoTokenizer

from arabert.preprocess import ArabertPreprocessor
from farasa.segmenter import FarasaSegmenter

from tqdm import tqdm
import numpy as np


# -----------------------------------------------CUSTOM CALLBACK CLASS-------------------------------------------------

class myCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, metrics, threshold):
        self.metrics = metrics
        self.threshold = float(threshold)

  
    def on_epoch_end(self, epoch, logs = {}):  
        if ( logs.get(self.metrics) >= self.threshold ):
            print("\n\nThreshold reached- {} : {}\nTraining Stopped.\n\n".format(self.metrics,self.threshold))
            self.model.stop_training = True

            



class auxillaryModel:
    
    def __init__(self, embedding_matrix, max_seq_len, num_classes):
        
        self.embedding_matrix = embedding_matrix
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.input_len = max_seq_len
        self.embedding_vocab = embedding_matrix.shape[0]
        self.embedding_dimension = embedding_matrix.shape[1]
        self.model_dict  = {

              "CNN_BiLSTM_e1" : self.__CNN_BiLSTM_e1__
          }

    # ------------------------------------SIMPLE FEED FORWARD NEURAL NETWORK-------------------------------------------------

    def FFNN (self, input_len, input_dimension, embedding_dimension, num_classes, 
              embedding_matrix , num_layers = 2, trainable = True, kr = None, br = None):

      # Define Sequential model
      model = Sequential()

      # Add embedding layer
      embedding_layer = model.add( Embedding(input_length = input_len, input_dim = input_dimension,
                                             output_dim = embedding_dimension, weights = [embedding_matrix], trainable = trainable) )
      model.add(GlobalMaxPooling1D())

      # Add hidden layers
      for num in range(num_layers + 4, 4, -1):
        n_neurons = 2**num  
        model.add( Dense(n_neurons, kernel_regularizer = kr, bias_regularizer = br, activation = 'relu') )

      # Add output layer
      model.add( Dense(num_classes, activation = 'softmax') )

      # Compile and return
      model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
      return model

    # ------------------------------------SIMPLE CONVOLUTIONAL NEURAL NETWORK-------------------------------------------------

    def CNN(self, input_len, input_dimension, embedding_dimension, num_classes, embedding_matrix, 
             kernel_size , num_layers = 1, dropout_rate = 0.0, kr = None, br = None, trainable = False):

        # Define Sequential Model
        model = Sequential()

        # Add embedding layer
        embedding_layer = model.add(Embedding (input_length = input_len, input_dim = input_dimension, 
                                               output_dim = embedding_dimension, weights = [embedding_matrix],
                                                trainable = trainable) )

        # Add convolutional layers
        for num in range(num_layers + 4, 4, -1):
            n_neurons = 2**num  
            model.add( Conv1D(n_neurons, kernel_size = kernel_size,
                              kernel_regularizer = kr, bias_regularizer = br,
                              activation  = "relu") )
            model.add( GlobalMaxPooling1D() )
            model.add( Dropout(dropout_rate) )

        # Add output layer
        model.add( Dense(num_classes, activation = "softmax") )

        # Compile and return
        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
        return model

    # ---------------------------------------SIMPLE BILSTM NEURAL NETWORK-------------------------------------------------

    def BILSTM_1(self, input_len, input_dimension, embedding_dimension, num_classes,
               embedding_matrix, layer1 = 128, layer2 = 64,  dropout_rate = 0.0, trainable = False):

        # Define Sequential Model
        model = Sequential()

        # Add embedding layer
        model.add( (Embedding(input_length = input_len, input_dim = input_dimension, output_dim = embedding_dimension,
                              weights = [embedding_matrix], trainable = trainable)) )

        # Add hidden layer
        model.add( (Bidirectional(LSTM (layer1 , dropout = dropout_rate, return_sequences = True))))
        model.add(GlobalMaxPooling1D() )
        model.add(Dense (layer2, activation = "relu"))

        # Add output layer
        model.add(Dense(num_classes, activation = "softmax"))

        # Compile and return
        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
        return model

    # -------------------------------------- BILSTM WITH TIMEDISTRIBURED LAYER-------------------------------------------------

    def BILSTM_2(self, input_len, input_dimension, embedding_dimension, num_classes,
                  embedding_matrix, dropout_rate = 0.5, trainable = False):

        # Define Sequential model
        model = Sequential()

        # Add embedding layer
        model.add(Embedding(input_length = input_len, input_dim = input_dimension, output_dim = embedding_dimension,
                            weights = [embedding_matrix], trainable = trainable))

        # Add hidden layers
        model.add(Bidirectional(LSTM(100, dropout = dropout_rate, return_sequences = True)))
        model.add(TimeDistributed (Dense(100, activation = "relu")))
        model.add(Flatten())
        model.add(Dense(100, activation = "relu"))

        # Add output layer
        model.add(Dense(num_classes, activation = "softmax"))

        # Compile and return
        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
        return model

    # -------------------------------------- CNN-LSTM HYBRID LAYER-------------------------------------------------
    def CNN_LSTM_e2(self, input_len, input_dimension, embedding_dimension, num_classes, 
              embedding_matrix, dropout_rate, kernel_size):
        model = tf.keras.models.Sequential()
        model.add(Embedding(input_dim=input_dimension, input_length = input_len, output_dim = embedding_dimension, weights=[embedding_matrix],
                            trainable=False))
        model.add(Conv1D(256, kernel_size = kernel_size, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Dropout(dropout_rate))
        model.add(LSTM(256,dropout=0.3))
        model.add(Dropout(dropout_rate))
        model.add(Dense(64, activation='relu'))
        #model.add(Dense(32, activation='relu'))
        model.add((Dense(num_classes, activation='softmax')))
        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
        return model


    # -------------------------------------------CNN-GRU STACKED--------------------------------------------------------
    def CNN_GRU_e1(self, input_len, input_dimension, embedding_dimension, num_classes, 
              embedding_matrix, dropout_rate, kernel_size):
      model = tf.keras.models.Sequential()
      model.add(Embedding(input_dim=input_dimension, input_length=input_len, output_dim = embedding_dimension, weights=[embedding_matrix],
                          trainable=False))
      model.add(Conv1D(64, kernel_size = kernel_size, activation='relu'))
      model.add(MaxPooling1D())
      model.add(Dropout(dropout_rate))
      model.add(GRU(256, return_sequences=False))
      model.add(Dense(64, activation='relu'))
      model.add(Dropout(dropout_rate))
      #model.add(Dense(16, activation='relu'))
      model.add((Dense(num_classes, activation='softmax')))
      model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['acc'])
      return model

    # -------------------------------------------CNN-BiLSTM v1--------------------------------------------------------
    # @staticmethod
    def __CNN_BiLSTM_e1__(self):

        custom_input = tf.keras.layers.Input(shape = (self.input_len, ))
        embedding_layer = tf.keras.layers.Embedding(input_length = self.input_len, input_dim = self.embedding_vocab,
                                            output_dim = self.embedding_dimension, weights = [self.embedding_matrix], trainable = False)(custom_input)

        Y = tf.keras.layers.Conv1D(filters = 256, kernel_size=3, padding = "same", activation="relu")(embedding_layer)
        Y = tf.keras.layers.MaxPooling1D()(Y)
        Y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, recurrent_dropout=0.2, return_sequences=True))(Y)
        Y = tf.keras.layers.Dropout(0.3)(Y)
        Y = tf.keras.layers.GlobalMaxPooling1D()(Y)
        model = tf.keras.Model(inputs = custom_input, outputs = Y)
        return model
    
    
    def load_model(self, model_name = "CNN_BiLSTM_e1"):
        
        model = self.model_dict[model_name]()
        return model
        
    

class myBERT:

  
    def __init__(self, model_name = "aubmindlab/bert-base-arabertv02" , tokenizer_name = "aubmindlab/bert-base-arabertv02", max_seq_len = 100):

          self.model_name = model_name
          self.tokenizer_name = tokenizer_name
          self.max_seq_len = max_seq_len


    """   Function to preprocess text based on the arabic tokenizer model.
          @param sentences: list/array of text sequences to be preprocessed
          @return sentence_preprocessed: preprocessed sentences  
    """ 

    def preprocess(self, sentences):
        
          farasa_segmenter = FarasaSegmenter(interactive=True)
          arabert_preprocessor = ArabertPreprocessor(model_name = self.model_name)

          sentences_preprocessed = [arabert_preprocessor.preprocess(x) for x in sentences]

          del farasa_segmenter
          del arabert_preprocessor

          return sentences_preprocessed


    """   Function to tokenize text based on the tokinizer model.
          @param sentences: list/array of text sequences to be tokenized
          @param tokenizer: Tokenizer model
          @return: np_array(input_ids, attention_masks, input_segments)   
    """ 
    def get_input_features(self, sentences, encode_plus = True, preprocess = True):

        max_len = self.max_seq_len
        input_ids, attention_masks, input_segments = [],[],[]

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, do_lower_case=False, do_basic_tokenize=True, never_split=True,
                                                 add_special_tokens=True, pad_to_max_length=True, max_length = self.max_seq_len)

        if preprocess:
          sentences = self.preprocess(sentences)

        for sentence in tqdm(sentences):
          if not encode_plus:
            sentence_encoded = tokenizer.encode(sentence, add_special_tokens=True, pad_to_max_length=True, max_length=self.max_seq_len,
                                                return_attention_mask=True, return_token_type_ids=True)
          else:
            sentence_encoded = tokenizer.encode_plus(sentence, add_special_tokens=True, pad_to_max_length=True, max_length=self.max_seq_len,
                                                  return_attention_mask=True, return_token_type_ids=True)

          input_ids.append(sentence_encoded['input_ids'])
          attention_masks.append(sentence_encoded['attention_mask'])
          input_segments.append(sentence_encoded['token_type_ids'])

        return [np.asarray(input_ids, dtype='int32'), np.asarray(attention_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')]



    """   Function to intitialize and configure the BERT model.
          @param sequence_classifier (type: boolean): Flag to indicate the type of transformer model
          @param num_labels (type: int):  number of output classes
          @return: transformer_model (type: TFAutoModel / TFAutoModelForSequenceClassification)
    """
    
    def load_model(self, num_labels, sequence_classifier = False):
        
        config = AutoConfig.from_pretrained(self.model_name)
        config.output_hidden_states = False
        config.num_labels = num_labels
        config.return_dict = False

        if not sequence_classifier:
          transformer_model = TFAutoModel.from_pretrained(self.model_name, config = config)
        else:
          transformer_model = TFAutoModelForSequenceClassification.from_pretrained(self.model_name, config = config)

        return transformer_model