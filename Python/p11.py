import numpy as np

def step(x):
    return np.where(x >= 0, 1, 0)

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([[0], [1], [1], [1]])

class Perceptron:
    def __init__(self, size, lr=0.1, epochs=1000):
        self.w = np.zeros((size, 1))
        self.b = 0
        self.lr = lr
        self.epochs = epochs

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                xi = xi.reshape(-1, 1)
                z = np.dot(xi.T, self.w) + self.b
                pred = step(z)
                err = yi - pred
                self.w += self.lr * err * xi
                self.b += self.lr * err

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return step(z)

p_and = Perceptron(size=2)
p_and.train(X_and, y_and)

p_or = Perceptron(size=2)
p_or.train(X_or, y_or)

print("AND Function Predictions:")
print(p_and.predict(X_and))

print("\nOR Function Predictions:")
print(p_or.predict(X_or))

print("\nAND Function Prediction for input [0, 0]:")
print(p_and.predict(np.array([[0, 0]])))
print("\nOR Function Prediction for input [0, 0]:")
print(p_or.predict(np.array([[0, 0]])))

print("\nAND Function Prediction for input [0, 1]:")
print(p_and.predict(np.array([[0, 1]])))
print("\nOR Function Prediction for input [0, 1]:")
print(p_or.predict(np.array([[0, 1]])))

print("\nAND Function Prediction for input [1, 0]:")
print(p_and.predict(np.array([[1, 0]])))
print("\nOR Function Prediction for input [1, 0]:")
print(p_or.predict(np.array([[1, 0]])))

print("\nAND Function Prediction for input [1, 1]:")
print(p_and.predict(np.array([[1, 1]])))
print("\nOR Function Prediction for input [1, 1]:")
print(p_or.predict(np.array([[1, 1]])))