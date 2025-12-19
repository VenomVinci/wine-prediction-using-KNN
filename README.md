---

# KNN Wine Classification

A K-Nearest Neighbors (KNN) classifier used to predict wine quality based on physicochemical features.

# Dataset

* **Source**: UCI Machine Learning Repository
* **Features**: Alcohol, pH, Sulphates, Citric Acid, etc.
* **Target**: Wine quality (0-10)

## Requirements

* Python 3.x
* Libraries: `numpy`, `pandas`, `scikit-learn`

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/knn-wine-classifier.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load and preprocess the dataset.
2. Train the KNN model:

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.neighbors import KNeighborsClassifier

   # Train-test split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   # KNN model
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(X_train, y_train)

   # Accuracy
   print("Accuracy:", knn.score(X_test, y_test))  # ~80%
   ```

## Results

* **Accuracy**: ~80%

---

