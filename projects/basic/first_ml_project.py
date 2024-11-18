"""
First ML Project: Iris Flower Classification
This template shows a complete ML project structure with best practices.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class IrisClassifier:
    def __init__(self):
        self.data = None
        self.target = None
        self.feature_names = None
        self.target_names = None
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load and prepare the Iris dataset"""
        iris = load_iris()
        self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.target = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
    def explore_data(self):
        """Basic data exploration"""
        print("Dataset Shape:", self.data.shape)
        print("\nFeature Statistics:")
        print(self.data.describe())
        
        # Visualize the data
        plt.figure(figsize=(10, 6))
        sns.pairplot(pd.concat([self.data, 
                              pd.Series(self.target, name='target')], axis=1), 
                    hue='target')
        plt.show()
    
    def prepare_data(self):
        """Prepare data for training"""
        X = self.scaler.fit_transform(self.data)
        return train_test_split(X, self.target, test_size=0.2, random_state=42)
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, 
                                 target_names=self.target_names))

def main():
    # Initialize the classifier
    classifier = IrisClassifier()
    
    # Load and explore data
    classifier.load_data()
    classifier.explore_data()
    
    # Prepare data and train model
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    classifier.train_model(X_train, y_train)
    
    # Evaluate the model
    classifier.evaluate_model(X_test, y_test)

if __name__ == "__main__":
    main()
