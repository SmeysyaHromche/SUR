from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

# Helper for model interpretation
# support: count of data
# recall: TP / (TP + FN)
# precision: TP / (TP + FP)
# F1-scroe: how good model (0: bad, 1: goat)


# C: [0, 1] light, [1, 10] hard
# class_weight: balacned 

class ImageBinaryClassifier:

    def __init__(self, n_components=0.9, C=0.8):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("svm", LinearSVC(
                C=C,
                class_weight="balanced",
                max_iter=10000,
                random_state=42
            ))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)