import pandas as pd
from sklearn.model_selection import train_test_split
import os
from convokit import Corpus, download
from datasets import load_dataset

def prepare_dataset():
    corpus = Corpus(filename=download("movie-corpus"))
    
    # Extract utterances from the corpus
    utterances_df = corpus.get_utterances_dataframe()
    
    # Load IMDB dataset for additional movie content
    imdb = load_dataset("imdb")
    imdb_df = pd.DataFrame(imdb['train'])
    
    # Combine datasets
    utterances_df['text'] = utterances_df['text']
    imdb_df['text'] = imdb_df['text']
    combined_df = pd.concat([
        utterances_df[['text']], 
        imdb_df[['text']], 
    ])
    
    # Filter for longer, more interesting responses
    combined_df = combined_df[combined_df['text'].str.len() > 50]
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(combined_df, test_size=0.1, random_state=42)
    
    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data = prepare_dataset()
    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
