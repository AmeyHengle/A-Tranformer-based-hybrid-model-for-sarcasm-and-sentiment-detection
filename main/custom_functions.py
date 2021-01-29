# Custom functions for data prepraration and inference generation

import pandas as pd
import numpy as np
import re
import string
from matplotlib import pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from farasa.diacratizer import FarasaDiacritizer
from farasa.segmenter import FarasaSegmenter
from farasa.stemmer import FarasaStemmer

STOP_WORDS = ['،', 'ء', 'ءَ', 'آ', 'آب', 'آذار', 'آض', 'آل', 'آمينَ', 'آناء', 'آنفا', 'آه', 'آهاً', 'آهٍ', 'آهِ', 'أ', 'أبدا', 'أبريل', 'أبو', 'أبٌ', 'أجل', 'أجمع', 'أحد', 'أخبر', 'أخذ', 'أخو', 'أخٌ', 'أربع', 'أربعاء', 'أربعة', 'أربعمئة', 'أربعمائة', 'أرى', 'أسكن', 'أصبح', 'أصلا', 'أضحى', 'أطعم', 'أعطى', 'أعلم', 'أغسطس', 'أفريل', 'أفعل به', 'أفٍّ', 'أقبل', 'أكتوبر', 'أل', 'ألا', 'ألف', 'ألفى', 'أم', 'أما', 'أمام', 'أمامك', 'أمامكَ', 'أمد', 'أمس', 'أمسى', 'أمّا', 'أن', 'أنا', 'أنبأ', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'أنتِ', 'أنشأ', 'أنه', 'أنًّ', 'أنّى', 'أهلا', 'أو', 'أوت', 'أوشك', 'أول', 'أولئك', 'أولاء', 'أولالك', 'أوّهْ', 'أى', 'أي', 'أيا', 'أيار', 'أيضا', 'أيلول', 'أين', 'أيّ', 'أيّان', 'أُفٍّ', 'ؤ', 'إحدى', 'إذ', 'إذا', 'إذاً', 'إذما', 'إذن', 'إزاء', 'إلى', 'إلي', 'إليكم', 'إليكما', 'إليكنّ', 'إليكَ', 'إلَيْكَ', 'إلّا', 'إمّا', 'إن', 'إنَّ', 'إى', 'إياك', 'إياكم', 'إياكما', 'إياكن', 'إيانا', 'إياه', 'إياها', 'إياهم', 'إياهما', 'إياهن', 'إياي', 'إيهٍ', 'ئ', 'ا', 'ا?', 'ا?ى', 'االا', 'االتى', 'ابتدأ', 'ابين', 'اتخذ', 'اثر', 'اثنا', 'اثنان', 'اثني', 'اثنين', 'اجل', 'احد', 'اخرى', 'اخلولق', 'اذا', 'اربعة', 'اربعون', 'اربعين', 'ارتدّ', 'استحال', 'اصبح', 'اضحى', 'اطار', 'اعادة', 'اعلنت', 'اف', 'اكثر', 'اكد', 'الآن', 'الألاء', 'الألى', 'الا', 'الاخيرة', 'الان', 'الاول', 'الاولى', 'التى', 'التي', 'الثاني', 'الثانية', 'الحالي', 'الذاتي', 'الذى', 'الذي', 'الذين', 'السابق', 'الف', 'اللاتي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'الماضي', 'المقبل', 'الوقت', 'الى', 'الي', 'اليه', 'اليها', 'اليوم', 'اما', 'امام', 'امس', 'امسى', 'ان', 'انبرى', 'انقلب', 'انه', 'انها', 'او', 'اول', 'اي', 'ايار', 'ايام', 'ايضا', 'ب', 'بؤسا', 'بإن', 'بئس', 'باء', 'بات', 'باسم', 'بان', 'بخٍ', 'بد', 'بدلا', 'برس', 'بسبب', 'بسّ', 'بشكل', 'بضع', 'بطآن', 'بعد', 'بعدا', 'بعض', 'بغتة', 'بل', 'بلى', 'بن', 'به', 'بها', 'بهذا', 'بيد', 'بين', 'بَسْ', 'بَلْهَ', 'ة', 'ت', 'تاء', 'تارة', 'تاسع', 'تانِ', 'تانِك', 'تبدّل', 'تجاه', 'تحت', 'تحوّل', 'تخذ', 'ترك', 'تسع', 'تسعة', 'تسعمئة', 'تسعمائة', 'تسعون', 'تسعين', 'تشرين', 'تعسا', 'تعلَّم', 'تفعلان', 'تفعلون', 'تفعلين', 'تكون', 'تلقاء', 'تلك', 'تم', 'تموز', 'تينك', 'تَيْنِ', 'تِه', 'تِي', 'ث', 'ثاء', 'ثالث', 'ثامن', 'ثان', 'ثاني', 'ثلاث', 'ثلاثاء', 'ثلاثة', 'ثلاثمئة', 'ثلاثمائة', 'ثلاثون', 'ثلاثين', 'ثم', 'ثمان', 'ثمانمئة', 'ثمانون', 'ثماني', 'ثمانية', 'ثمانين', 'ثمنمئة', 'ثمَّ', 'ثمّ', 'ثمّة', 'ج', 'جانفي', 'جدا', 'جعل', 'جلل', 'جمعة', 'جميع', 'جنيه', 'جوان', 'جويلية', 'جير', 'جيم', 'ح', 'حاء', 'حادي', 'حار', 'حاشا', 'حاليا', 'حاي', 'حبذا', 'حبيب', 'حتى', 'حجا', 'حدَث', 'حرى', 'حزيران', 'حسب', 'حقا', 'حمدا', 'حمو', 'حمٌ', 'حوالى', 'حول', 'حيث', 'حيثما', 'حين', 'حيَّ', 'حَذارِ', 'خ', 'خاء', 'خاصة', 'خال', 'خامس', 'خبَّر', 'خلا', 'خلافا', 'خلال', 'خلف', 'خمس', 'خمسة', 'خمسمئة', 'خمسمائة', 'خمسون', 'خمسين', 'خميس', 'د', 'دال', 'درهم', 'درى', 'دواليك', 'دولار', 'دون', 'دونك', 'ديسمبر', 'دينار', 'ذ', 'ذا', 'ذات', 'ذاك', 'ذال', 'ذانك', 'ذانِ', 'ذلك', 'ذهب', 'ذو', 'ذيت', 'ذينك', 'ذَيْنِ', 'ذِه', 'ذِي', 'ر', 'رأى', 'راء', 'رابع', 'راح', 'رجع', 'رزق', 'رويدك', 'ريال', 'ريث', 'رُبَّ', 'ز', 'زاي', 'زعم', 'زود', 'زيارة', 'س', 'ساء', 'سابع', 'سادس', 'سبت', 'سبتمبر', 'سبحان', 'سبع', 'سبعة', 'سبعمئة', 'سبعمائة', 'سبعون', 'سبعين', 'ست', 'ستة', 'ستكون', 'ستمئة', 'ستمائة', 'ستون', 'ستين', 'سحقا', 'سرا', 'سرعان', 'سقى', 'سمعا', 'سنة', 'سنتيم', 'سنوات', 'سوف', 'سوى', 'سين', 'ش', 'شباط', 'شبه', 'شتانَ', 'شخصا', 'شرع', 'شمال', 'شيكل', 'شين', 'شَتَّانَ', 'ص', 'صاد', 'صار', 'صباح', 'صبر', 'صبرا', 'صدقا', 'صراحة', 'صفر', 'صهٍ', 'صهْ', 'ض', 'ضاد', 'ضحوة', 'ضد', 'ضمن', 'ط', 'طاء', 'طاق', 'طالما', 'طرا', 'طفق', 'طَق', 'ظ', 'ظاء', 'ظل', 'ظلّ', 'ظنَّ', 'ع', 'عاد', 'عاشر', 'عام', 'عاما', 'عامة', 'عجبا', 'عدا', 'عدة', 'عدد', 'عدم', 'عدَّ', 'عسى', 'عشر', 'عشرة', 'عشرون', 'عشرين', 'عل', 'علق', 'علم', 'على', 'علي', 'عليك', 'عليه', 'عليها', 'علًّ', 'عن', 'عند', 'عندما', 'عنه', 'عنها', 'عوض', 'عيانا', 'عين', 'عَدَسْ', 'غ', 'غادر', 'غالبا', 'غدا', 'غداة', 'غير', 'غين', 'ـ', 'ف', 'فإن', 'فاء', 'فان', 'فانه', 'فبراير', 'فرادى', 'فضلا', 'فقد', 'فقط', 'فكان', 'فلان', 'فلس', 'فهو', 'فو', 'فوق', 'فى', 'في', 'فيفري', 'فيه', 'فيها', 'ق', 'قاطبة', 'قاف', 'قال', 'قام', 'قبل', 'قد', 'قرش', 'قطّ', 'قلما', 'قوة', 'ك', 'كأن', 'كأنّ', 'كأيّ', 'كأيّن', 'كاد', 'كاف', 'كان', 'كانت', 'كانون', 'كثيرا', 'كذا', 'كذلك', 'كرب', 'كسا', 'كل', 'كلتا', 'كلم', 'كلَّا', 'كلّما', 'كم', 'كما', 'كن', 'كى', 'كيت', 'كيف', 'كيفما', 'كِخ', 'ل', 'لأن', 'لا', 'لا سيما', 'لات', 'لازال', 'لاسيما', 'لام', 'لايزال', 'لبيك', 'لدن', 'لدى', 'لدي', 'لذلك', 'لعل', 'لعلَّ', 'لعمر', 'لقاء', 'لكن', 'لكنه', 'لكنَّ', 'للامم', 'لم', 'لما', 'لمّا', 'لن', 'له', 'لها', 'لهذا', 'لهم', 'لو', 'لوكالة', 'لولا', 'لوما', 'ليت', 'ليرة', 'ليس', 'ليسب', 'م', 'مئة', 'مئتان', 'ما', 'ما أفعله', 'ما انفك', 'ما برح', 'مائة', 'ماانفك', 'مابرح', 'مادام', 'ماذا', 'مارس', 'مازال', 'مافتئ', 'ماي', 'مايزال', 'مايو', 'متى', 'مثل', 'مذ', 'مرّة', 'مساء', 'مع', 'معاذ', 'معه', 'مقابل', 'مكانكم', 'مكانكما', 'مكانكنّ', 'مكانَك', 'مليار', 'مليم', 'مليون', 'مما', 'من', 'منذ', 'منه', 'منها', 'مه', 'مهما', 'ميم', 'ن', 'نا', 'نبَّا', 'نحن', 'نحو', 'نعم', 'نفس', 'نفسه', 'نهاية', 'نوفمبر', 'نون', 'نيسان', 'نيف', 'نَخْ', 'نَّ', 'ه', 'هؤلاء', 'ها', 'هاء', 'هاكَ', 'هبّ', 'هذا', 'هذه', 'هل', 'هللة', 'هلم', 'هلّا', 'هم', 'هما', 'همزة', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هي', 'هيا', 'هيهات', 'هيّا', 'هَؤلاء', 'هَاتانِ', 'هَاتَيْنِ', 'هَاتِه', 'هَاتِي', 'هَجْ', 'هَذا', 'هَذانِ', 'هَذَيْنِ', 'هَذِه', 'هَذِي', 'هَيْهات', 'و', 'و6', 'وأبو', 'وأن', 'وا', 'واحد', 'واضاف', 'واضافت', 'واكد', 'والتي', 'والذي', 'وان', 'واهاً', 'واو', 'واوضح', 'وبين', 'وثي', 'وجد', 'وراءَك', 'ورد', 'وعلى', 'وفي', 'وقال', 'وقالت', 'وقد', 'وقف', 'وكان', 'وكانت', 'ولا', 'ولايزال', 'ولكن', 'ولم', 'وله', 'وليس', 'ومع', 'ومن', 'وهب', 'وهذا', 'وهو', 'وهي', 'وَيْ', 'وُشْكَانَ', 'ى', 'ي', 'ياء', 'يفعلان', 'يفعلون', 'يكون', 'يلي', 'يمكن', 'يمين', 'ين', 'يناير', 'يوان', 'يورو', 'يوليو', 'يوم', 'يونيو', 'ّأيّان']




# Reading dataframe
def read_data(path):
  data = pd.read_csv(path)
  data = data.fillna('')
  return data


# Extract all stopwords from a text file and store them in a list.
def get_stopwords(textfile):
  stopwords = []
  file = open(textfile, "r", encoding = "utf-8")
  for line in file:
    line = line.replace("\n","")
    stopwords.append(line)
  file.close()
  return stopwords


# Clean hashtags, mentions, special/unwnated/unprintable characters from the tweet. 
 
def clean_text(text): 
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

  # arabic_diacritics = re.compile("""
  #                            ّ    | # Shadda
  #                            َ    | # Fatha
  #                            ً    | # Tanwin Fath
  #                            ُ    | # Damma
  #                            ٌ    | # Tanwin Damm
  #                            ِ    | # Kasra
  #                            ٍ    | # Tanwin Kasr
  #                            ْ    | # Sukun
  #                            ـ     # Tatwil/Kashida
  #                        """, re.VERBOSE)

  punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

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
  # text = re.sub(arabic_diacritics, '', text)

  # #longation
  # text = re.sub("[إأآا]", "ا", text)
  # text = re.sub("ى", "ي", text)
  # text = re.sub("ؤ", "ء", text)
  # text = re.sub("ئ", "ء", text)
  # text = re.sub("ة", "ه", text)
  # text = re.sub("گ", "ك", text)

  # #stopwords
  # text = ' '.join(word for word in text.split() if word not in STOP_WORDS)

  #emojis and emoticons
  #-----
  

  return text

#Preprocess text using Farasa 
def preprocess(text):
  segmenter = FarasaSegmenter()
  diacratizer = FarasaDiacritizer()
  stemmer = FarasaStemmer()

  text = segmenter.segment(text)
  text = diacratizer.diacritize(text)
  text = stemmer.stem(text)
  
  return text


# Preprocess Emojis
def preprocess_emojis(emoji_mapping, tweet):
    processed_tweet = ""
    
    for character in tweet:
        if character in emoji_mapping:
#             print(character, "--->", emoji_mapping[character])
            processed_tweet = processed_tweet + emoji_mapping[character] + " "
        else:
            processed_tweet = processed_tweet + character

    for word in tweet:
        if word in emoji_mapping:
            processed_tweet = processed_tweet.replace(word, emoji_mapping[word] + " ")
    
    return processed_tweet

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

# def get_word_embeddings(filepath, vocab, embedding_dimension):

#   word_embeddings = np.zeros((len(vocab) + 1, embedding_dimension))
# #   word_embeddings = []
#   embedding_file = open(filepath, "r", encoding = "utf8")
#   count = 0

#   for line in embedding_file:
#     try:
#         line = line.split()
#         word = line[0]
#         if word in vocab:
#           word_vector = np.asarray(line[1:], dtype = "float32")
#           if len(word_vector) == embedding_dimension:
#             word_embeddings[vocab[word]] = word_vector
#           else:
#             print('\nVector size does not match with embedding_dimension:\t', word_vector)
#           count+=1
    
#     except exception as e:
#         print("Exception:",e,"\n\n",line)
  
#   print("Total word embeddings read: {}\n".format(count))
#   embedding_file.close()
#   return word_embeddings 


def get_word_embeddings(filepath, vocab, ft = False, save_embeddings = False):
  
  binary = False
  embedding_dimension = 0
  embedding_dict = {}

  if ft == True:
    word_vectors = fasttext.load_model(filepath)
    embedding_dimension = len(get_word_vector(list(word_vectors.get_words())[0]))
    print("File loaded. Total Words: {},\t Embedding Dimension: {}".format(len(word_vectors.get_words()), embedding_dimension))

    for word in vocab:
      try:
        wv = word_vectors.get_word_vector[word]
        embedding_dict[word] = wv

      except Exception as e:
        print("Exception reading vector for word:  {}, \n Exception : {} \n".format(word, e))
        continue

    print("Total embeddings found: {}\n\n".format(len(embedding_dict)))

  else:
    if ".bin" in filepath :
      print("Processing binary file")
      binary = True
      
    print("Loading vectors from: {} \n".format(filepath))
    word_vectors =  KeyedVectors.load_word2vec_format(filepath,binary=binary)

    embedding_dimension = len(list(word_vectors.vectors)[0])
    print("File loaded. Total Words: {},\t Embedding Dimension: {}".format(len(word_vectors.vocab), embedding_dimension))
  
  for word in vocab:
    try:
      wv = word_vectors.wv[word]
      embedding_dict[word] = wv

    except Exception as e:
      print("Exception reading vector for word:  {}, \n Exception : {} \n".format(word, e))
      continue

  print("Total embeddings found: {}\n\n".format(len(embedding_dict)))

  if save_embeddings == True:
    output_file = filepath.split('.')[0] + '.pkl'
    with open(output_file, 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Word embeddings saved to {} \n\n".format(output_file))

  return embedding_dict


def get_embedding_matrix(vocab, embedding_dict_file = "", embedding_dict = {}, embedding_dimension = 100):
  except_count = 0

  if not embedding_dict:
    try:
        print("Loading embeddings from: ",embedding_dict_file)
        with open(embedding_dict_file, "rb") as f:
            embedding_dict = pickle.load(f)
    except Exception as e:
        print("\nException: ",e)
        
#   vocab_size = len(embedding_dict.keys()) + 1
  vocab_size = len(vocab) + 1
  embedding_matrix = np.zeros((vocab_size, embedding_dimension))


  for i, word in enumerate(vocab):
    try:
      embedding_matrix[i] = embedding_dict[word]
    except Exception as e:
#       print("Exception reading vector for word:  {}, \n Exception : {} \n".format(word, e))
      except_count += 1
      continue

  print("\nTotal words processed: {}".format(len(embedding_matrix) - except_count))
  print("Words not found: ", except_count)
    
  return embedding_matrix



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


def load_from_pickle(filepath):
    file = open(filepath, "r",encoding = "utf8")
    data = pickle.load(file)
    return data
    
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
