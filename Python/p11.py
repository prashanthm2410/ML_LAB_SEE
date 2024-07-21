import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

def create_and_train_model(inputs, labels, epochs=500, learning_rate=0.01):
    model = Sequential([
        Dense(1, input_dim=2, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(inputs, labels, epochs=epochs, verbose=0)
    return model

model_and = create_and_train_model(X_and, y_and)
model_or = create_and_train_model(X_or, y_or)


def test_model(model, inputs):
    predictions = model.predict(inputs)
    predictions = [round(pred[0]) for pred in predictions]
    return predictions

print("AND Function Predictions:")
print(test_model(model_and, X_and))

print("\nOR Function Predictions:")
print(test_model(model_or, X_or))

and_test_input = np.array([[1, 1]])
or_test_input = np.array([[0, 1]])

print("\nAND Function Prediction for input [1, 1]:")
print(test_model(model_and, and_test_input))

print("\nOR Function Prediction for input [0, 1]:")
print(test_model(model_or, or_test_input))