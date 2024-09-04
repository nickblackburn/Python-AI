import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Ensure types are correct and handle missing values
    df['Data Value'] = pd.to_numeric(df['Data Value'], errors='coerce')
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
    df.dropna(subset=['Data Value', 'Geo Place Name', 'Start_Date'], inplace=True)
    
    # Add 'Day_Of_Year' feature
    df['Day_Of_Year'] = df['Start_Date'].dt.dayofyear
    
    # Convert 'Start_Date' to a numerical representation (e.g., timestamp or ordinal)
    df['Start_Date'] = df['Start_Date'].astype(np.int64) // 10**9  # Convert to seconds
    
    return df


def encode_categorical(df_train, df_test):
    # Convert categorical variables to numerical using label encoding
    label_encoder = preprocessing.LabelEncoder()
    df_train['Geo Place Name'] = label_encoder.fit_transform(df_train['Geo Place Name'])
    df_test['Geo Place Name'] = label_encoder.transform(df_test['Geo Place Name'])
    return df_train, df_test


def scale_features(X_train, X_test):
    # Separate numerical and date features for training set
    num_features_train = X_train.select_dtypes(include=[np.number])
    date_features_train = X_train.select_dtypes(include=[np.datetime64])

    # Separate numerical and date features for test set
    num_features_test = X_test.select_dtypes(include=[np.number])
    date_features_test = X_test.select_dtypes(include=[np.datetime64])

    # Scale numerical features for training set
    scaler = preprocessing.StandardScaler().fit(num_features_train)
    num_features_scaled_train = scaler.transform(num_features_train)

    # Impute NaN values in numerical features for both training and test sets
    imputer = SimpleImputer(strategy='mean')
    num_features_scaled_train = imputer.fit_transform(num_features_scaled_train)
    num_features_scaled_test = imputer.transform(scaler.transform(num_features_test))

    # Combine scaled numerical features with date features for training set
    X_train_scaled = pd.concat([pd.DataFrame(num_features_scaled_train, columns=num_features_train.columns), date_features_train], axis=1)

    # Combine scaled numerical features with date features for test set
    X_test_scaled = pd.concat([pd.DataFrame(num_features_scaled_test, columns=num_features_test.columns), date_features_test], axis=1)

    # Check and handle NaN values after imputation
    nan_cols_train = X_train_scaled.columns[X_train_scaled.isna().any()]
    nan_cols_test = X_test_scaled.columns[X_test_scaled.isna().any()]

    X_train_scaled = X_train_scaled.dropna(subset=nan_cols_train)
    X_test_scaled = X_test_scaled.dropna(subset=nan_cols_test)

    # Print NaN values after imputation
    print("NaN values in X_train_scaled after imputation:\n", X_train_scaled.isna().sum())
    print("NaN values in X_test_scaled after imputation:\n", X_test_scaled.isna().sum())

    return X_train_scaled, X_test_scaled


def evaluate_model(clf, X_test, y_test):
    yhat = clf.predict(X_test)
    print("Mean Absolute Error: {:.4f}".format(mean_absolute_error(y_test, yhat)))
    print("Mean Squared Error: {:.4f}".format(mean_squared_error(y_test, yhat)))
    print("R2-Score: {:.4f}".format(r2_score(y_test, yhat)))


def plot_3d_heatmap(X, y, feature1='Geo Place Name', feature2='Day_Of_Year'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[feature1], X[feature2], y, c=y, cmap='viridis', marker='o')

    ax.set_xlabel(feature1)
    ax.set_ylabel(feature2)
    ax.set_zlabel('Data Value')
    ax.set_title("3D Heatmap")

    plt.show()


def batch_process_print(df, batch_size=1000):
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        # Perform your batch processing steps here
        print(f"Processed {min(i+batch_size, len(df))} rows out of {len(df)}")


def main():
    path = "/Users/nicholas/Desktop/Machine Learning/data/Air_Quality.csv"
    air_quality_df = load_data(path)
    air_quality_df = preprocess_data(air_quality_df)

    # Split data into training and test sets
    train_size = int(len(air_quality_df) * 0.8)
    train_df, test_df = air_quality_df[:train_size], air_quality_df[train_size:]

    # Encode categorical features
    train_df, test_df = encode_categorical(train_df, test_df)

    feature_columns = ['Geo Place Name', 'Start_Date', 'Day_Of_Year']

    X_train = train_df[feature_columns]
    y_train = train_df['Data Value']

    X_test = test_df[feature_columns]
    y_test = test_df['Data Value']

    print("Before preprocessing: X_train shape =", X_train.shape, "y_train shape =", y_train.shape)
    print("X_train columns:", X_train.columns)
    print("X_test columns:", X_test.columns)

    X_train, X_test = scale_features(X_train, X_test)

    print("After scaling and concatenation: X_train shape =", X_train.shape, "y_train shape =", y_train.shape)
    print("X_train columns:", X_train.columns)
    print("X_test columns:", X_test.columns)

    # Perform randomized search for hyperparameter tuning
    param_dist = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
    n_iter_search = 4  # Adjusted to 4 iterations
    random_search = RandomizedSearchCV(SVR(), param_distributions=param_dist, n_iter=n_iter_search, verbose=2, cv=5)
    random_search.fit(X_train, y_train)

    print("Best parameters found: ", random_search.best_params_)

    clf_best = random_search.best_estimator_

    # Batch process and print
    batch_process_print(air_quality_df)

    # 3D Heatmap Visualization
    plot_3d_heatmap(X_test, y_test)

    # Evaluate the model
    print("Metrics for Best SVM Model:")
    evaluate_model(clf_best, X_test, y_test)


if __name__ == "__main__":
    main()

