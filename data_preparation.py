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

    # Load astronomy dataset
    astronomy = load_dataset("spaceflights/astronomy_dataset")
    astronomy_df = pd.DataFrame(astronomy['train'])

    # Load astrology dataset (hypothetical, replace with actual dataset if available)
    # astrology = load_dataset("your_astrology_dataset")
    # astrology_df = pd.DataFrame(astrology['train'])

    # Combine datasets
    movie_df['text'] = movie_df['utterance']
    imdb_df['text'] = imdb_df['text']
    astronomy_df['text'] = astronomy_df['text']  # Adjust column name if necessary
    # astrology_df['text'] = astrology_df['text']  # Uncomment when you have the astrology dataset

    combined_df = pd.concat([
        movie_df[['text']], 
        imdb_df[['text']], 
        astronomy_df[['text']],
        # astrology_df[['text']]  # Uncomment when you have the astrology dataset
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
