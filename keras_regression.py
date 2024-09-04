import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load data
path = "/Users/nicholas/Desktop/Machine Learning/data/concrete_data.csv"
concrete_data = pd.read_csv(path)

# Separate predictors and target
predictors = concrete_data.drop('Strength', axis=1)
target = concrete_data['Strength']

# Normalize predictors using StandardScaler
scaler = StandardScaler()
predictors_norm = scaler.fit_transform(predictors)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)

# Define regression model
def regression_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Build the model
model = regression_model()

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=2, callbacks=[early_stopping])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
