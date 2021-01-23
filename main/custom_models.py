# Custom model architectures for training deep neural networks

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPooling1D, MaxPooling1D, Conv1D, Dropout, GlobalAveragePooling1D,LSTM, Bidirectional, TimeDistributed, Flatten


# -----------------------------------------------CUSTOM CALLBACK CLASS-------------------------------------------------

class myCallbacks(tf.keras.callbacks.Callback):

    def __init__(self, metrics, threshold):
        self.metrics = metrics
        self.threshold = float(threshold)

  
    def on_epoch_end(self, epoch, logs = {}):  
        if ( logs.get(self.metrics) >= self.threshold ):
            print("\n\nThreshold reached- {} : {}\nTraining Stopped.\n\n".format(self.metrics,self.threshold))
            self.model.stop_training = True

            
# ------------------------------------SIMPLE FEED FORWARD NEURAL NETWORK-------------------------------------------------

def FFNN (input_length, input_dimension, embedding_dimension, output_dimension, 
          embedding_matrix , num_layers = 2, trainable = True, kr = None, br = None):
  
  # Define Sequential model
  model = Sequential()

  # Add embedding layer
  embedding_layer = model.add( Embedding(input_length = input_length, input_dim = input_dimension,
                                         output_dim = embedding_dimension, weights = [embedding_matrix], trainable = trainable) )
  model.add(GlobalMaxPooling1D())
  
  # Add hidden layers
  for num in range(num_layers + 4, 4, -1):
    n_neurons = 2**num  
    model.add( Dense(n_neurons, kernel_regularizer = kr, bias_regularizer = br, activation = 'relu') )

  # Add output layer
  model.add( Dense(output_dimension, activation = 'softmax') )

  # Compile and return
  model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
  return model

# ------------------------------------SIMPLE CONVOLUTIONAL NEURAL NETWORK-------------------------------------------------

def CNN(input_length, input_dimension, embedding_dimension, output_dimension, embedding_matrix, 
         kernel_size , num_layers = 1, dropout_rate = 0.0, kr = None, br = None, trainable = False):
    
    # Define Sequential Model
    model = Sequential()
    
    # Add embedding layer
    embedding_layer = model.add(Embedding (input_length = input_length, input_dim = input_dimension, 
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
    model.add( Dense(output_dimension, activation = "softmax") )
    
    # Compile and return
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
    return model

# ---------------------------------------SIMPLE BILSTM NEURAL NETWORK-------------------------------------------------

def BILSTM(input_length, input_dimension, embedding_dimension, output_dimension,
           embedding_matrix, layer1 = 128, layer2 = 64,  dropout_rate = 0.0, trainable = False):
    
    # Define Sequential Model
    model = Sequential()
    
    # Add embedding layer
    model.add( (Embedding(input_length = input_length, input_dim = input_dimension, output_dim = embedding_dimension,
                          weights = [embedding_matrix], trainable = trainable)) )
    
    # Add hidden layer
    model.add( (Bidirectional(LSTM (layer1 , dropout = dropout_rate, return_sequences = True))))
    model.add(GlobalMaxPooling1D() )
    model.add(Dense (layer2, activation = "relu"))
    
    # Add output layer
    model.add(Dense(output_dimension, activation = "softmax"))
    
    # Compile and return
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
    return model

# -------------------------------------- BILSTM WITH TIMEDISTRIBURED LAYER-------------------------------------------------

def BILSTM_2(input_length, input_dimension, embedding_dimension, output_dimension,
              embedding_matrix, dropout_rate = 0.5, trainable = False):
    
    # Define Sequential model
    model = Sequential()
    
    # Add embedding layer
    model.add(Embedding(input_length = input_length, input_dim = input_dimension, output_dim = embedding_dimension,
                        weights = [embedding_matrix], trainable = trainable))
    
    # Add hidden layers
    model.add(Bidirectional(LSTM(100, dropout = dropout_rate, return_sequences = True)))
    model.add(TimeDistributed (Dense(100, activation = "relu")))
    model.add(Flatten())
    model.add(Dense(100, activation = "relu"))
    
    # Add output layer
    model.add(Dense(output_dimension, activation = "softmax"))
    
    # Compile and return
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
    return model

# -------------------------------------- CNN-LSTM HYBRID LAYER-------------------------------------------------
def CNN_LSTM(input_length, input_dimension, embedding_dimension, output_dimension, 
          embedding_matrix, dropout_rate, kernel_size):

    model = Sequential()

    model.add(Embedding(input_dim=input_dimension, input_length=input_length, output_dim = embedding_dimension, weights=[embedding_matrix],
                        trainable=False))
                        
    model.add(Conv1D(128, kernel_size = kernel_size, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(256, dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add((Dense(output_dimension, activation='softmax')))
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["acc"])
    return model