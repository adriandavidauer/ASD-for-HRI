'''
Train Models using hdf5 data or with PAZ pipeline from the fullsize Dataset
'''

# System imports
import argparse
from pathlib import Path
import pickle
import glob
import os

# 3rd party imports
import h5py
from keras.models import load_model, clone_model
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import callbacks
from tqdm import tqdm

# local imports
from .asd import NormalizeShapeSample, GetShapeFeatures

# end file header
__author__      = 'Adrian Auer'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path", help="path to a folder holding pickle files or hdf5 files", type=str)
    parser.add_argument("-f", "--feature_type", help="Type to use to train the model.", choices=["LipShape", "FaceShape"], type=str, default="LipShape")
    parser.add_argument("-s", "--seed", help="Seed for random number generator for reproducibility", type=int, default=42)
    parser.add_argument("-e", "--epochs", help="Number of epochs to train the model", type=int, default=500)
    args = parser.parse_args()
    # set seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f"Training model for {args.feature_type} with data from {args.path}")
    if args.feature_type == "LipShape":
        model = clone_model(load_model(str(Path(__file__).absolute().parent.parent / "models" / 'lipFeatureModel.keras')))
    elif args.feature_type == "FaceShape":
        model = clone_model(load_model(str(Path(__file__).absolute().parent.parent / "models" / 'faceFeatureModel.keras')))


    normalize = NormalizeShapeSample()
    getShapeFeatures = GetShapeFeatures(architecture=args.feature_type)
    
    if args.path.endswith('.hdf5') or args.path.endswith('.h5'):
        source = "hdf5"
        print("Training model with hdf5 data")
        data = hdf5_file = h5py.File(args.path, mode='r')
        # # print structure of hdf5 file 
        # def print_attrs(name, obj):
        #     print(f'{name}: {obj}')
        # data.visititems(print_attrs)
        
        # data needs to be normalized
        
        normalized_training_data = np.empty(data['x_train'].shape)
        print("Preparing training data...")
        for i, sample in tqdm(enumerate(data['x_train']), total=data['x_train'].shape[0]):
            normalized_training_data[i] = normalize([sample])[0] # no batch calculation implemented for NormalizeShapeSample, so we need to add an extra dimension and then remove it again after normalization
        y_train = data['y_train'][:]


    else:
        print("Training model with PAZ pipeline data")
        source = "paz"
        #  set up dataset
        normalized_training_data = []
        y_train = []
        #  glob through all folders with pickle files and load them
        files = glob.glob(os.path.join(os.path.join(args.path, '**'), '*.pickle'))
        print("Preparing training data...")
        for samplePath in tqdm(files, total=len(files)):
            with open(samplePath, 'rb') as file:
                sample = pickle.load(file)
                sample_frames = []
                for frame in sample['data']:
                    sample_frames.append(getShapeFeatures(frame))
                sample_frames = np.array(sample_frames)
                normalized_training_data.append(normalize([sample_frames])[0]) # no batch calculation implemented for NormalizeShapeSample, so we need to add an extra dimension and then remove it again after normalization
                y_train.append(sample['label'])
        normalized_training_data = np.array(normalized_training_data)
        y_train = np.array(y_train)


    # generate data for training and validation
    X_train, X_val, Y_train, Y_val = train_test_split(normalized_training_data, y_train, test_size=0.2, random_state=42)
    # set up callbacks for tensorboard, reduce rl on plateau, early stopping and model checkpointing
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                            patience=10, min_lr=0.001, cooldown=2, mode='max')
    earlyStopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True, mode='max')

    checkpoint = callbacks.ModelCheckpoint(f"{source}_{args.feature_type}_{{val_accuracy:.4f}}.keras", monitor='val_accuracy', save_best_only=True, mode='max')

    tensorboard = callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=args.epochs, batch_size=32, callbacks=[reduce_lr, earlyStopping, checkpoint, tensorboard])
