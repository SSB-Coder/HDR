import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

def train_initial_model():
    print("--- Initializing Training Process ---")
    print("Fetching MNIST dataset... (This might take a minute)")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    
    X = X / 255.0
    y = y.astype(int) 
    X, y = shuffle(X, y, random_state=42)

    print("Training the initial model on 70,000 images...")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        verbose=True, 
        max_iter=60,
        random_state=42
    )

    mlp.fit(X, y) 
    
    joblib.dump(mlp, 'mnist_model.joblib')
    print("SUCCESS: mnist_model.joblib created safely.")

if __name__ == "__main__":
    train_initial_model()