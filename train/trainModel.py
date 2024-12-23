# trainModel.py
# Created by pongwsl on 23 Dec 2024
# Last updated 23 Dec 2024
# Script to train a CNN model to detect pinching actions

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def loadData(filePath):
    """Load data from CSV file."""
    if not os.path.exists(filePath):
        print(f"Error: {filePath} does not exist.")
        return None, None
    data = pd.read_csv(filePath)
    # Ensure the last column is the label
    X = data.iloc[:, :-1].values  # All columns except last
    y = data.iloc[:, -1].values   # Last column
    return X, y

def preprocessData(X, y):
    """Preprocess data: encode labels and normalize features."""
    # Encode labels
    yEncoded = np.where(y == 'on', 1, 0)
    
    # Handle missing values if any
    if np.isnan(X).any():
        print("Warning: Missing values detected. Removing incomplete samples.")
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        yEncoded = yEncoded[mask]
    
    # Normalize features
    scaler = StandardScaler()
    Xscaled = scaler.fit_transform(X)
    
    # Reshape for CNN: (samples, timesteps, features)
    # Here, treat landmarks as timesteps and x,y,z as features
    Xreshaped = Xscaled.reshape(Xscaled.shape[0], 21, 3)
    
    # One-hot encode labels
    yCategorical = to_categorical(yEncoded, num_classes=2)
    
    return Xreshaped, yCategorical, scaler

def createCNNmodel(inputShape):
    """Create a 1D CNN model."""
    model = Sequential([
        Conv1D(64, kernel_size = 3, activation = 'relu', input_shape = inputShape),
        MaxPooling1D(pool_size = 2),
        Dropout(0.3),
        Conv1D(128, kernel_size = 3, activation = 'relu'),
        MaxPooling1D(pool_size = 2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation = 'relu'),
        Dropout(0.3),
        Dense(2, activation = 'softmax')
    ])
    
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def trainModel(X, y, k=5):
    """Train CNN model using k-fold cross-validation."""
    skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)
    foldNo = 1
    accPerFold = []
    lossPerFold = []
    
    for trainIndex, testIndex in skf.split(X, np.argmax(y, axis = 1)):
        print(f'Training for fold {foldNo} ...')
        
        Xtrain, Xtest = X[trainIndex], X[testIndex]
        yTrain, yTest = y[trainIndex], y[testIndex]
        
        model = createCNNmodel(input_shape = (X.shape[1], X.shape[2]))
        
        # Early stopping
        earlyStop = EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            Xtrain, yTrain,
            epochs = 100,
            batch_size = 32,
            validation_data = (Xtest, yTest),
            callbacks = [earlyStop],
            verbose = 0
        )
        
        # Evaluate
        scores = model.evaluate(Xtest, yTest, verbose=0)
        print(f'Score for fold {foldNo}: {model.metrics_names[0]} of {scores[0]:.4f}; {model.metrics_names[1]} of {scores[1]*100:.2f}%')
        accPerFold.append(scores[1] * 100)
        lossPerFold.append(scores[0])
        foldNo += 1
    
    print('---')
    print('Score per fold')
    for i in range(len(accPerFold)):
        print(f'> Fold {i+1} - Loss: {lossPerFold[i]:.4f} - Accuracy: {accPerFold[i]:.2f}%')
    print('---')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(accPerFold):.2f}% (+- {np.std(accPerFold):.2f}%)')
    print(f'> Loss: {np.mean(lossPerFold):.4f}')
    
    return model  # Return the last trained model

def main():
    # Load data
    X, y = loadData('data.csv')
    if X is None or y is None:
        return
    
    # Preprocess data
    Xprocessed, yProcessed, scaler = preprocessData(X, y)
    
    # Train model with k-fold cross-validation
    model = trainModel(Xprocessed, yProcessed, k=5)
    
    # Save the trained model and scaler
    model.save('pinchModel.h5')
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully as 'pinchModel.h5' and 'scaler.pkl'.")

if __name__ == "__main__":
    main()