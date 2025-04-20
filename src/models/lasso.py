import numpy as np

class LassoRegression():
    def __init__(self, iterations=10000):
        self.iterations=iterations
        pass

    def fit(self, X, y, reg, lr=0.01):
        self.M, self.N = X.shape
        self.B = np.zeros(self.N)
        self.B_0 = 0
        self.X = X
        self.y = y
        self.reg = reg
        self.lr = lr

        for k in range(self.iterations):
            self.gradient_descent_update()

        return self

    def gradient_descent_update(self):
        yhat = self.predict(self.X)
        gradB = np.zeros(self.N)

        for j in range(self.N):
            if self.B[j] > 0:
                gradB[j] = (-2 * (self.X[:,j]).dot(self.y - yhat) + self.reg) / self.M
            else:
                gradB[j] = (-2 * (self.X[:,j]).dot(self.y - yhat) - self.reg) / self.M
        
        gradB_0 = -2 * np.sum(self.y - yhat) / self.M

        self.B = self.B - self.lr*gradB
        self.B_0 = self.B_0 - self.lr*gradB_0
        return self

    def predict(self, X):
        return(X.dot(self.B) + self.B_0)

