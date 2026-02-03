"""
Main training and evaluation pipeline for Web Traffic Analysis.
Supports both baseline LSTM and enhanced Seq2Seq models.
"""

import argparse
import torch
from pathlib import Path
import sys

from model.params import get_params, get_data_config
from model.input_pipe import DataProcessor, create_dataloaders
from model.model import get_model, get_loss_function
from model.trainer import Trainer, create_optimizer
from model.eval import evaluate_model, Visualizer


def train_baseline(args):
    """Train baseline LSTM model."""
    print("\n" + "="*60)
    print("BASELINE LSTM TRAINING")
    print("="*60 + "\n")
    
    # Load parameters
    params = get_params("baseline")
    data_config = get_data_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Initialize data processor
    processor = DataProcessor(data_config)
    
    # Load and prepare training data
    print("Loading training data...")
    train_df = processor.load_data(data_config.train_csv)
    train_series, train_timestamps = processor.prepare_time_series(
        train_df,
        chunk_time=params.chunk_time,
        feature=params.feature
    )
    print(f"Training series shape: {train_series.shape}")
    
    # Split data
    train_data, val_data, _ = processor.split_data(
        train_series,
        train_split=params.train_split
    )
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Split timestamps
    train_size = len(train_data)
    timestamps_train = train_timestamps[:train_size]
    timestamps_val = train_timestamps[train_size:]
    
    # Normalize data
    train_scaled, val_scaled = processor.normalize_data(
        train_data,
        val_data,
        scaler_type=params.scaler_type
    )
    
    # Save scaler
    scaler_path = Path(params.save_dir) / 'scaler.pkl'
    processor.save_scaler(str(scaler_path))
    print(f"Saved scaler to {scaler_path}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_scaled,
        val_scaled,
        params,
        enhanced=False
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nBuilding model...")
    model = get_model(params, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss
    optimizer = create_optimizer(model, params)
    criterion = get_loss_function(loss_type=params.loss_type)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        params=params,
        save_dir=params.save_dir
    )
    
    # Train
    print("\nStarting training...\n")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    Visualizer.plot_training_history(
        history,
        save_path=str(Path(params.save_dir) / 'training_history.png')
    )
    
    # Load best model for evaluation
    trainer.load_checkpoint('best_model.pth')
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60 + "\n")
    
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        scaler=processor.scaler,
        spike_thresholds=[1.5, 2.0, 2.5],
        spike_tolerance=params.spike_tolerance,
        save_dir=str(Path(params.save_dir) / 'val_results')
    )
    
    # Evaluate on test set if available
    if Path(data_config.test_csv).exists():
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60 + "\n")
        
        # Load test data
        test_df = processor.load_data(data_config.test_csv)
        test_series, test_timestamps = processor.prepare_time_series(
            test_df,
            chunk_time=params.chunk_time,
            feature=params.feature
        )
        print(f"Test series shape: {test_series.shape}")
        
        # Normalize test data
        test_scaled = processor.scaler.transform(test_series.reshape(-1, 1))
        
        # Create test dataloader
        from model.input_pipe import TimeSeriesDataset
        from torch.utils.data import DataLoader
        
        test_ds = TimeSeriesDataset(
            test_scaled,
            window_size=params.train_window,
            predict_window=params.predict_window
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=params.batch_size,
            shuffle=False
        )
        
        # Evaluate
        test_results = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            scaler=processor.scaler,
            spike_thresholds=[1.5, 2.0, 2.5],
            spike_tolerance=params.spike_tolerance,
            save_dir=str(Path(params.save_dir) / 'test_results')
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Checkpoints saved to: {params.save_dir}")
    print("="*60 + "\n")


def train_enhanced(args):
    """Train enhanced Seq2Seq model."""
    print("\n" + "="*60)
    print("ENHANCED SEQ2SEQ TRAINING")
    print("="*60 + "\n")
    
    # Load parameters
    params = get_params("enhanced")
    data_config = get_data_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Initialize data processor
    processor = DataProcessor(data_config)
    
    # Load and prepare training data
    print("Loading training data...")
    train_df = processor.load_data(data_config.train_csv)
    train_series, train_timestamps = processor.prepare_time_series(
        train_df,
        chunk_time=params.chunk_time,
        feature=params.feature
    )
    print(f"Training series shape: {train_series.shape}")
    
    # Split data
    train_data, val_data, _ = processor.split_data(
        train_series,
        train_split=params.train_split
    )
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Split timestamps
    train_size = len(train_data)
    timestamps_train = train_timestamps[:train_size]
    timestamps_val = train_timestamps[train_size:]
    
    # Normalize data
    train_scaled, val_scaled = processor.normalize_data(
        train_data,
        val_data,
        scaler_type=params.scaler_type
    )
    
    # Save scaler
    scaler_path = Path(params.save_dir) / 'scaler.pkl'
    processor.save_scaler(str(scaler_path))
    print(f"Saved scaler to {scaler_path}")
    
    # Create dataloaders with features
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_scaled,
        val_scaled,
        params,
        enhanced=True,
        timestamps_train=timestamps_train,
        timestamps_val=timestamps_val
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nBuilding model...")
    model = get_model(params, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and loss
    optimizer = create_optimizer(model, params)
    criterion = get_loss_function(loss_type='smape')  # Use SMAPE for enhanced model
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        params=params,
        save_dir=params.save_dir
    )
    
    # Train
    print("\nStarting training...\n")
    history = trainer.train(train_loader, val_loader, early_stopping_patience=15)
    
    # Plot training history
    Visualizer.plot_training_history(
        history,
        save_path=str(Path(params.save_dir) / 'training_history.png')
    )
    
    # Load best model for evaluation
    trainer.load_checkpoint('best_model.pth')
    
    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60 + "\n")
    
    val_results = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        scaler=processor.scaler,
        spike_thresholds=[1.5, 2.0, 2.5],
        spike_tolerance=params.spike_tolerance,
        save_dir=str(Path(params.save_dir) / 'val_results')
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Checkpoints saved to: {params.save_dir}")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Web Traffic Analysis - Training and Evaluation Pipeline"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='baseline',
        choices=['baseline', 'enhanced', 'both'],
        help='Model type to train: baseline (LSTM), enhanced (Seq2Seq), or both'
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing before training'
    )
    
    args = parser.parse_args()
    
    # Run preprocessing if requested
    if args.preprocess:
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60 + "\n")
        
        from preprocessing.data_parser import load_raw_data, preprocess_data, save_processed_data
        
        raw_train_path = "dataset/train.txt"
        raw_test_path = "dataset/test.txt"
        processed_train_path = "dataset/train.csv"
        processed_test_path = "dataset/test.csv"
        
        raw_train = load_raw_data(raw_train_path)
        raw_test = load_raw_data(raw_test_path)
        print(f"Loaded {len(raw_train)} train, {len(raw_test)} test records")

        train_df, train_failed = preprocess_data(raw_train)
        test_df, test_failed = preprocess_data(raw_test)
        print(f"Processed data shape:\n- Train: {train_df.shape}\n- Test: {test_df.shape}")
        
        # Save processed data
        print(f"\nSaving processed data...")
        save_processed_data(train_df, processed_train_path)
        save_processed_data(test_df, processed_test_path)
        print("Preprocessing complete!\n")
    
    # Train models
    if args.model == 'baseline':
        train_baseline(args)
    elif args.model == 'enhanced':
        train_enhanced(args)
    elif args.model == 'both':
        train_baseline(args)
        train_enhanced(args)


if __name__ == "__main__":
    main()