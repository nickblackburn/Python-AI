import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, log_loss
import seaborn as sns

# Load data
path = "/Users/nicholas/Desktop/Machine Learning/data/ChurnData.csv"
churn_df = pd.read_csv(path)

# Feature selection and type conversion
selected_features = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']
churn_df = churn_df[selected_features + ['churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

# Feature scaling
X = StandardScaler().fit_transform(churn_df[selected_features])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, churn_df['churn'], test_size=0.2, random_state=4)

# Logistic Regression
C_param = 0.01
solver_param = 'liblinear'

LR = LogisticRegression(C=C_param, solver=solver_param).fit(X_train, y_train)

# Predictions
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

# Model evaluation
print("Jaccard Score:", jaccard_score(y_test, yhat, pos_label=0))
conf_matrix = confusion_matrix(y_test, yhat, labels=[1, 0])
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, yhat))

# Log Loss
print("Log Loss:", log_loss(y_test, yhat_prob))

# Plot confusion matrix using seaborn
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['churn=1', 'churn=0'], yticklabels=['churn=1', 'churn=0'])
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
