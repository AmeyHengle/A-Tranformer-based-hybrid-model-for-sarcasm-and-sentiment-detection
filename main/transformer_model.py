from transformers import TFDistilBertModel
from transformers import TFAutoModel
from transformers import TFDistilBertForSequenceClassification, DistilBertConfig
from transformers import AutoConfig, TFAutoModelForSequenceClassification
from transformers import DistilBertTokenizer, RobertaTokenizer, AutoTokenizer

from arabert.preprocess import ArabertPreprocessor
from farasa.segmenter import FarasaSegmenter

from tqdm import tqdm
import numpy as np

# model_name = "aubmindlab/bert-base-arabertv02"
# tokenizer_name = "aubmindlab/bert-base-arabertv02"
# max_seq_len_arabert = 100

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