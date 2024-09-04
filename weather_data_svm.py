import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Ensure types are correct and handle missing values
    df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce')
    df['Sunshine'] = pd.to_numeric(df['Sunshine'], errors='coerce')
    df['Temp9am'] = pd.to_numeric(df['Temp9am'], errors='coerce')
    df.dropna(subset=['Rainfall', 'Sunshine', 'Temp9am', 'RainTomorrow'], inplace=True)
    return df

def encode_categorical(df):
    # Convert categorical variables to numerical using label encoding
    # Note: You might want to use one-hot encoding for WindGustDir, WindDir9am, WindDir3pm if needed
    label_encoder = preprocessing.LabelEncoder()
    df['RainTomorrow'] = label_encoder.fit_transform(df['RainTomorrow'])
    return df

def scale_features(X_train, X_test):
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def evaluate_model(clf, X_test, y_test):
    yhat = clf.predict(X_test)
    print("Accuracy Score: {:.4f}".format(accuracy_score(y_test, yhat)))
    print("Jaccard Index: {:.4f}".format(jaccard_score(y_test, yhat)))
    print("F1-Score: {:.4f}".format(f1_score(y_test, yhat)))
    print("Mean Absolute Error: {:.4f}".format(mean_absolute_error(y_test, yhat)))
    print("Mean Squared Error: {:.4f}".format(mean_squared_error(y_test, yhat)))
    print("R2-Score: {:.4f}".format(r2_score(y_test, yhat)))

def plot_3d_scatter(X, y, feature1='Rainfall', feature2='Sunshine', feature3='Temp9am'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[X['RainTomorrow']==0][feature1], X[X['RainTomorrow']==0][feature2], X[X['RainTomorrow']==0][feature3], c='b', marker='o', label='No')
    ax.scatter(X[X['RainTomorrow']==1][feature1], X[X['RainTomorrow']==1][feature2], X[X['RainTomorrow']==1][feature3], c='r', marker='^', label='Yes')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel(feature3)
    ax.set_title("3D Scatter Plot")
    ax.legend()

    plt.show()

def batch_process_print(df, batch_size=500):
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        # Perform your batch processing steps here
        print(f"Processed {min(i+batch_size, len(df))} rows out of {len(df)}")

def main():
    path = "/Users/nicholas/Desktop/Machine Learning/data/Weather_Data.csv"
    weather_df = load_data(path)
    weather_df = preprocess_data(weather_df)
    weather_df = encode_categorical(weather_df)

    feature_columns = ['Rainfall', 'Sunshine', 'Temp9am']

    X = weather_df[feature_columns]
    y = weather_df['RainTomorrow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train, X_test = scale_features(X_train, X_test)

    # Perform grid search for hyperparameter tuning
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    clf_best = grid_search.best_estimator_

    # Batch process and print
    batch_process_print(weather_df)

    # Convert X_test to a Pandas DataFrame before using join
    X_test_df = pd.DataFrame(X_test, columns=feature_columns)
    
    # 3D Scatter Plot
    plot_3d_scatter(X_test_df.join(y_test), y_test)

    # Evaluate the model
    print("Metrics for Best SVM Model:")
    evaluate_model(clf_best, X_test, y_test)

if __name__ == "__main__":
    main()
