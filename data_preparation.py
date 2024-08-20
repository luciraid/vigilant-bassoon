import pandas as pd
from sklearn.model_selection import train_test_split
import os

from convokit import Corpus, download

def prepare_dataset():
    corpus = Corpus(filename=download("movie-corpus"))

    # Extract data from the corpus using convokit
    utterances = corpus.get_utterances_dataframe()

    # ... (Rest of your data preparation logic)

    return train_data, val_data

    # Load Cornell Movie-Dialogs Corpus
    movie_dialogs = load_dataset("movie_corpus")
    movie_df = pd.DataFrame(movie_dialogs['train'])

    # Load IMDB dataset for additional movie content
    imdb = load_dataset("imdb")
    imdb_df = pd.DataFrame(imdb['train'])

    # Load astronomy dataset
    # Note: This dataset might not exist. You may need to find an alternative or create it.
    try:
        astronomy = load_dataset("spaceflights/astronomy_dataset")
        astronomy_df = pd.DataFrame(astronomy['train'])
    except Exception as e:
        print(f"Error loading astronomy dataset: {e}")
        astronomy_df = pd.DataFrame()

    # Astrology dataset is not available in Hugging Face datasets
    # You'll need to provide this dataset separately

    # Combine datasets
    movie_df['text'] = movie_df['utterance']
    imdb_df['text'] = imdb_df['text']
    
    if not astronomy_df.empty:
        astronomy_df['text'] = astronomy_df['text']  # Adjust column name if necessary
    
    combined_df = pd.concat([
        movie_df[['text']], 
        imdb_df[['text']], 
        astronomy_df[['text']] if not astronomy_df.empty else pd.DataFrame()
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
