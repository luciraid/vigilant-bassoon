import pandas as pd
from sklearn.model_selection import train_test_split
import os
from convokit import Corpus, download
import psutil
import time

def log_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def prepare_dataset(sample_size=10000, chunk_size=5000, save_interval=10000):
    start_time = time.time()
    log_memory_usage()

    print("Downloading and loading movie corpus...")
    corpus = Corpus(filename=download("movie-corpus"))
    
    print("Extracting utterances from the corpus...")
    utterances_df = corpus.get_utterances_dataframe()
    
    print("Preparing dataset...")
    utterances_df = utterances_df[['text']]
    
    print("Sampling data...")
    utterances_df = utterances_df.sample(n=min(sample_size, len(utterances_df)), random_state=42)
    
    train_chunks = []
    val_chunks = []
    
    print("Processing data in chunks...")
    for i, chunk in enumerate(pd.read_csv(utterances_df, chunksize=chunk_size)):
        chunk = chunk[chunk['text'].str.len().gt(50)]
        train, val = train_test_split(chunk, test_size=0.1, random_state=42)
        train_chunks.append(train)
        val_chunks.append(val)
        
        if (i + 1) * chunk_size % save_interval == 0:
            print(f"Saving intermediate results (chunk {i})...")
            pd.concat(train_chunks).to_csv(f'train_data_chunk_{i}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'val_data_chunk_{i}.csv', index=False)
        
        log_memory_usage()
    
    print("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)
    
    log_memory_usage()
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data = prepare_dataset()
    print("Saving final results...")
    train_data.to_csv('train_data.csv', index=False)
    val_data.to_csv('val_data.csv', index=False)
    print("Process completed.")
