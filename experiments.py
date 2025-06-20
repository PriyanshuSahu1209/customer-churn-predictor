import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

# Load and preprocess data
data = pd.read_csv('data/Churn_Modelling.csv')

# Encode Gender
label_encoder_gender = LabelEncoder()
data["Gender"] = label_encoder_gender.fit_transform(data["Gender"])

# One-hot encode Geography
onehot_encoder_geo = OneHotEncoder(sparse_output=False, drop='first')
onehot_encoder_geo.fit(data[["Geography"]])
geo_encoded = onehot_encoder_geo.transform(data[["Geography"]])
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))
data = pd.concat([data.drop("Geography", axis=1), geo_df], axis=1)

# Save encoders
with open("models/label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)
with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(onehot_encoder_geo, f)

# Feature and target
X = data.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
y = data["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.values)
X_test = scaler.transform(X_test.values)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Build ANN model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=callbacks)

# Save model
model.save("model.h5")
