from preprocessing.data_parser import load_raw_data, preprocess_data, save_processed_data


if __name__ == "__main__":
    raw_data_path = "dataset/train.txt"
    processed_data_path = "dataset/train.csv"
    
    raw_lines = load_raw_data(raw_data_path)
    print(f"Loaded {len(raw_lines)} lines")

    df, failed = preprocess_data(raw_lines)
    print(f"Processed data shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nInformation:")
    print(df.info())
    
    # Save processed data
    print(f"\nSaving processed data to {processed_data_path}...")
    save_processed_data(df, processed_data_path)
    print("Ingestion pipeline completed successfully!\n")

    for l in failed[:5]:
        print(l)