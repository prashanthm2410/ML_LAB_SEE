import numpy as np

X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_not = np.array([[0], [0], [1], [0]])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.rand(input_size, hidden_size)
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b1 = np.random.rand(1, hidden_size)
        self.b2 = np.random.rand(1, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.W1) + self.b1)
        self.output = self.sigmoid(np.dot(self.hidden, self.W2) + self.b2)
        return self.output

    def backward(self, X, y):
        output_error = y - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        self.W2 += self.hidden.T.dot(output_delta)
        self.W1 += X.T.dot(hidden_delta)
        self.b2 += np.sum(output_delta, axis=0, keepdims=True)
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

mlp_and_not = MLP(input_size=2, hidden_size=4, output_size=1)
mlp_and_not.train(X_and_not, y_and_not, epochs=5000)

mlp_xor = MLP(input_size=2, hidden_size=4, output_size=1)
mlp_xor.train(X_xor, y_xor, epochs=5000)

print("AND-NOT Function Predictions:")
print(mlp_and_not.predict(X_and_not))

print("\nXOR Function Predictions:")
print(mlp_xor.predict(X_xor))

print("\nAND Function Prediction for input [0, 0]:")
print(mlp_and_not.predict(np.array([[0, 0]])))
print("\nOR Function Prediction for input [0, 0]:")
print(mlp_xor.predict(np.array([[0, 0]])))

print("\nAND Function Prediction for input [0, 1]:")
print(mlp_and_not.predict(np.array([[0, 1]])))
print("\nOR Function Prediction for input [0, 1]:")
print(mlp_xor.predict(np.array([[0, 1]])))

print("\nAND Function Prediction for input [1, 0]:")
print(mlp_and_not.predict(np.array([[1, 0]])))
print("\nOR Function Prediction for input [1, 0]:")
print(mlp_xor.predict(np.array([[1, 0]])))

print("\nAND Function Prediction for input [1, 1]:")
print(mlp_and_not.predict(np.array([[1, 1]])))
print("\nOR Function Prediction for input [1, 1]:")
print(mlp_xor.predict(np.array([[1, 1]])))
