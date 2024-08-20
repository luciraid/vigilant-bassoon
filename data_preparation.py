def prepare_astronomy_dataset():
    start_time = time.time()
    logging.info("Preparing astronomy datasets...")

    # Download datasets
    exoplanet_url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=csv"
    ads_url = "https://ui.adsabs.harvard.edu/api/v1/search/query?q=*&fl=id,title,author,date,citation_count,abstract&rows=10000&start=0"

    with ThreadPoolExecutor() as executor:
        exoplanet_df = executor.submit(download_and_preprocess, exoplanet_url, "exoplanet_data.csv").result()
        ads_df = executor.submit(download_and_preprocess, ads_url, "ads_data.csv").result()

    # Combine all datasets
    astronomy_df = pd.concat([exoplanet_df, ads_df], ignore_index=True)

    if len(astronomy_df) == 0:
        logging.warning("Astronomy datasets are empty. Skipping.")
        return None, None

    # Sample data
    logging.info("Sampling data...")
    astronomy_df = astronomy_df.sample(n=min(SAMPLE_SIZE, len(astronomy_df)), random_state=42)

    # Process data in chunks and save intermediate results
    train_chunks = []
    val_chunks = []
    logging.info("Processing data in chunks...")
    num_chunks = len(astronomy_df) // CHUNK_SIZE + 1

    for i, chunk in enumerate(pd.read_csv(astronomy_df, chunksize=CHUNK_SIZE)):
        train, val = split_dataset(chunk)
        train_chunks.append(train)
        val_chunks.append(val)

        if (i + 1) * CHUNK_SIZE % SAVE_INTERVAL == 0:
            logging.info(f"Saving intermediate results (chunk {i+1} of {num_chunks})...")
            pd.concat(train_chunks).to_csv(f'astronomy_train_data_chunk_{i+1}.csv', index=False)
            pd.concat(val_chunks).to_csv(f'astronomy_val_data_chunk_{i+1}.csv', index=False)

        logging.info(f"Processing chunk {i+1} of {num_chunks}")

    # Concatenate final results
    logging.info("Concatenating final results...")
    train_data = pd.concat(train_chunks)
    val_data = pd.concat(val_chunks)

    logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return train_data, val_data
