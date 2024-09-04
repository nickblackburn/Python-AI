import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, GridSearchCV
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
    df.loc[:, 'BareNuc'] = pd.to_numeric(df['BareNuc'], errors='coerce').astype('int')
    return df


def scale_features(X_train, X_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def evaluate_model(clf, X_test, y_test):
    yhat = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, yhat))

    cm = confusion_matrix(y_test, yhat, labels=[2, 4])
    df_cm = pd.DataFrame(cm, index=['Benign(2)', 'Malignant(4)'], columns=['Benign(2)', 'Malignant(4)'])

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    f1 = f1_score(y_test, yhat, average='weighted')
    jaccard = jaccard_score(y_test, yhat, pos_label=2)
    print("F1 Score: {:.4f}".format(f1))
    print("Jaccard Score: {:.4f}".format(jaccard))


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_3d_scatter(X, y, feature1=0, feature2=1, feature3=5):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Convert class labels to integers
    y = y.astype(int)

    # Get indices for Benign class (label = 2)
    benign_indices = np.where(y == 2)[0]
    ax.scatter(X[benign_indices, feature1], X[benign_indices, feature2], X[benign_indices, feature3], c='b', marker='o', label='Benign(2)')

    # Get indices for Malignant class (label = 4)
    malignant_indices = np.where(y == 4)[0]
    ax.scatter(X[malignant_indices, feature1], X[malignant_indices, feature2], X[malignant_indices, feature3], c='r', marker='^', label='Malignant(4)')

    ax.set_xlabel(f'Feature {feature1}')
    ax.set_ylabel(f'Feature {feature2}')
    ax.set_zlabel(f'Feature {feature3}')
    ax.set_title("3D Scatter Plot")
    ax.legend()

    plt.show()


def main():
    path = "/Users/nicholas/Desktop/Machine Learning/data/cell_samples.csv"
    cell_df = load_data(path)
    cell_df = preprocess_data(cell_df)

    feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
    X = np.asarray(feature_df)
    y = np.asarray(cell_df['Class'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train, X_test = scale_features(X_train, X_test)

    # Perform grid search for hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=3)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    
    clf_best = grid_search.best_estimator_

    # Visualize the learning curve
    title = "Learning Curves (SVM)"
    plot_learning_curve(clf_best, title, X_train, y_train, cv=3)
    plt.show()

    # Convert class labels to integers
    y_test = y_test.astype(int)

    # 3D Scatter Plot
    plot_3d_scatter(X_test, y_test)

    # Evaluate the model
    print("Metrics for Best SVM Model:")
    evaluate_model(clf_best, X_test, y_test)


if __name__ == "__main__":
    main()
