🍎 ML_learnings_KNN_multi_classific
Multi-Class Fruit Classification using K-Nearest Neighbors (KNN)
This project implements a K-Nearest Neighbors (KNN) multi-class classifier to identify different fruits based on their mass, dimensions, and color score.
This is part of my ML learning journey, and serves as my first complete end-to-end classification model.

📂 Dataset Overview

Dataset file: fruit.csv

The dataset contains 59 rows and 6 features:

Column	Description
fruit_label	Numeric class label
fruit_name	Actual fruit name
mass	Weight of fruit (grams)
width	Width (cm)
height	Height (cm)
color_score	Numeric color intensity (0–1 scale)
🔧 If your dataset contains the extra fruit_subtype column:
df = df.drop(columns=["fruit_subtype"])

🧠 Model: K-Nearest Neighbors (KNN)

Multi-class classifier

Distance metric: Euclidean

Tunable hyperparameter k

Non-parametric, simple, effective

Works well for small datasets

Model training happens inside Knn.ipynb.

📁 Project Structure
ML_learnings_KNN_multi_classific/
│
├── Knn.ipynb          # Main Jupyter notebook
├── fruit.csv          # Dataset used for classification
├── README.md          # Documentation
└── .gitignore         # (optional) Ignoring unnecessary files

🚀 Installation

Install required libraries:

pip install numpy pandas matplotlib scikit-learn

▶️ How to Run
Run the Notebook
jupyter notebook Knn.ipynb

Or convert to Python script
jupyter nbconvert --to script Knn.ipynb
python Knn.py

📈 Key Code Snippets
Train/Test Split
from sklearn.model_selection import train_test_split

X = df[['mass', 'width', 'height', 'color_score']]
y = df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

Training the KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

Prediction
pred = knn.predict(X_test)

Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

🎯 Results

Typical results achievable:

High accuracy (dataset is small but clean)

KNN performs well for separable classes

Visualizations show good clustering of fruit types

You can experiment with:

Scaling features

Changing k

Trying different distance metrics

🧪 Future Improvements

Feature normalization or standardization

Trying SVM, Decision Trees, or Logistic Regression

Adding more fruit types

Deploying as a small web app

Creating a Streamlit UI for predictions

🤝 Contributions

This repo is part of my ML learning journey.
Feel free to fork, improve, and extend it!

📜 License

Open for educational and practice use.
