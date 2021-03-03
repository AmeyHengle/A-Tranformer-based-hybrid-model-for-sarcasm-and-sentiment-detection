import torch
import pandas as pd
import numpy as np

from custom_functions import preprocess_emojis, clean_special_characters, find_emojis
from arabert.preprocess import ArabertPreprocessor


def preprocess(data):
    arabert_prep = ArabertPreprocessor(model_name.split("/")[-1])

    #ADD RELATIVE PATHS HERE
    filepath1 = "/content/test_data.csv"
    filepath2 = "/content/emoticons_mapping.csv"
    filepath3 = '/content/arabic-sentiment-lexicons.csv'

    # Processing emojis
    emoji_df = pd.read_csv(filepath2)
    emoji_mappings = {x : y for x,y in zip(emoji_df.emoticon.values.tolist(), emoji_df.arabic_translation.values.tolist())}
    print("\nProcessing emojis\nTotal emojis in dictionary: ", len(emoji_mappings))

    data["tweet_emoji_processed"] = data["tweet"].apply(lambda x : preprocess_emojis(emoji_mappings, x))

    print("\nDone")

    #Cleaning and Normalizing tweets
    data["tweet_cleaned"] = data["tweet_emoji_processed"].apply(lambda x: clean_special_characters(x))

    #Last check for emojis 
    data['tweet_temp'] = data['tweet_cleaned'].str.replace(r'[\U00010000-\U0010ffff]', find_emojis)

    #Apply Arabert Preprocessor
    data["tweet_preprocessed"] = data["tweet_temp"].apply(lambda x : arabert_prep.preprocess(x))
    return data 

