import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=100, reg_lambda=0.01, reg_type='l2'):
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type

    def softmax(self, z):
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        Y_onehot = np.eye(n_classes)[y]
        self.loss_history = []

        for epoch in range(self.epochs):
            logits = X @ self.W + self.b
            probs  = self.softmax(logits)
            error  = probs - Y_onehot
            dW = (X.T @ error) / n_samples
            db = error.mean(axis=0)

            if self.reg_type == 'l2':
                dW += self.reg_lambda * self.W
            elif self.reg_type == 'l1':
                dW += self.reg_lambda * np.sign(self.W)

            self.W -= self.lr * dW
            self.b -= self.lr * db

            loss = -np.mean(np.log(probs[np.arange(n_samples), y] + 1e-9))
            self.loss_history.append(loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:>3} | Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.softmax(X @ self.W + self.b), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)