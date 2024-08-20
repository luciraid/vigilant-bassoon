import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_dataset():
    # Load Cornell Movie-Dialogs Corpus
    movie_dialogs = load_dataset("cornell_movie_dialog_corpus")
    movie_df = pd.DataFrame(movie_dialogs['train'])

    # Load IMDB dataset for additional movie content
    imdb = load_dataset("imdb")
    imdb_df = pd.DataFrame(imdb['train'])

    # Combine datasets
    movie_df['text'] = movie_df['text_1'] + ' ' + movie_df['text_2']
    imdb_df['text'] = imdb_df['text']

    combined_df = pd.concat([movie_df[['text']], imdb_df[['text']]])

    # Split into train and validation sets
    train_data, val_data = train_test_split(combined_df, test_size=0.1, random_state=42)

    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data = prepare_dataset()
    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
