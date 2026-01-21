from preprocessing.data_parser import load_raw_data, preprocess_data, save_processed_data


if __name__ == "__main__":
    raw_train_path = "dataset/train.txt"
    raw_test_path = "dataset/test.txt"
    processed_train_path = "dataset/train.csv"
    processed_test_path = "dataset/test.csv"
    
    raw_train = load_raw_data(raw_train_path)
    raw_test = load_raw_data(raw_test_path)
    print(f"Loaded {len(raw_train)} train, {len(raw_test)} test")

    train_df, train_failed = preprocess_data(raw_train)
    test_df, test_failed = preprocess_data(raw_test)
    print(f"Processed data shape:\n- Train: {train_df.shape}\n- Test: {test_df.shape}")
    
    # Save processed data
    print(f"\nSaving processed data...")
    save_processed_data(train_df, processed_train_path)
    save_processed_data(test_df, processed_test_path)
    print("Complete parsing!")