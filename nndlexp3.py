 
# # Anirbaan Ghatak
# # C026
# # Aim: Implement a three-layer feedforward neural network for the IRIS dataset, including pre-processing, backpropagation, and tuning learning rates
# # and iterations to find optimal performance.


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score



data = pd.read_csv('Iris.csv')

 
# STEP2


species_encoder = OneHotEncoder()
species_encoded = species_encoder.fit_transform(data[['Species']]).toarray()
species_encoded_df = pd.DataFrame(species_encoded, columns=species_encoder.get_feature_names_out(['Species']))


data = data.drop('Species', axis=1)
data = pd.concat([data, species_encoded_df], axis=1)


data


feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])


X = data.drop(columns=species_encoded_df.columns)
y = species_encoded_df


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


tf.random.set_seed(42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(2, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='sigmoid'),
    tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
])



model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


import matplotlib.pyplot as plt

# Visualize accuracy and MSE during training
history = model.fit(X_train, y_train, epochs=510, batch_size=32, verbose=2, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")

y_pred_classes = y_pred.argmax(axis=1)
y_true_classes = y_test.argmax(axis=1)
accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"Accuracy: {accuracy}")



# Visualize accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Visualize MSE
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()


learning_rate = 0.01
iterations = 5000

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

mse_history = []
accuracy_history = []

for epoch in range(iterations):
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

    mse_history.append(history.history['loss'][0])
    accuracy_history.append(history.history['accuracy'][0])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{iterations} - MSE: {mse_history[-1]}, Accuracy: {accuracy_history[-1]}")

# Plot MSE and accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(mse_history)
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Epochs')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()



learning_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
iterations_list = [500, 5000]
results_df = pd.DataFrame(columns=['Learning Rate', 'Iterations', 'MSE', 'Accuracy'])

# Loop through different combinations of learning rates and iterations
for learning_rate in learning_rates:
    for iterations in iterations_list:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='sigmoid', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Dense(2, activation='sigmoid'),
            tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')
        ])


        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=iterations, batch_size=32, verbose=0)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        y_true_classes = y_test.argmax(axis=1)
        y_pred_classes = y_pred.argmax(axis=1)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)

        # Append the results to the DataFrame
        results_df = results_df.append({'Learning Rate': learning_rate,'Iterations': iterations,'MSE': mse,'Accuracy': accuracy }, ignore_index=True)



print(results_df)


