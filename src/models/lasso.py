import numpy as np

class LassoRegression():
    """Class to implement LASSO regression, based on (3.52) in Hastie's 'Elements of Statistical Learning' (Springer, 2013).
    The model is trained using gradient descent. 
    """

    def __init__(self, iterations=10000):
        self.iterations=iterations
        pass


    def fit(self, X, y, reg, lr=0.01):
        """Function to fit LASSO model, using gradient descent on the intercept and feature parameters.
        
        Parameters
        ------
        X : numpy.ndarray : column data
        y : numpy.ndrarry : target data
        reg : float : regularization term
        lr : float : learning rate 
        """

        self.M, self.N = X.shape
        self.B = np.zeros(self.N)
        self.B_0 = 0
        self.X = np.array(X)
        self.y = np.array(y)
        self.reg = reg
        self.lr = lr

        print("Fitting model.")
        print(f"Previous values: \nB_0: {self.B_0} \nB: {self.B}")

        for k in range(self.iterations):
            if k % 1000 == 0:
                print(f"Iteration: {k}/{self.iterations}")  # give a status update every 100 iterations
                self.gradient_descent_update(status=True)
            else:
                self.gradient_descent_update()

        print(f"Updated values: \nB_0: {self.B_0} \nB: {self.B}")
        return self


    def gradient_descent_update(self, status=False):
        """Function to perform gradient-descent updates on the LASSO parameters. 

        Parameters
        ----------
        status : bool : whether to print a status update on the parameter values and MSE in a given iteration.
        """

        yhat = self.predict(self.X)
        gradB = np.zeros(self.N)
        residual = np.subtract(self.y, yhat)
        
        if status:
            print(f"MSE: {(np.square(residual)).mean()}")

        for j in range(self.N):
            if self.B[j] > 0:
                gradB[j] = (-2 * np.dot(self.X[:,j], residual) + self.reg) / self.M
            else:
                gradB[j] = (-2 * np.dot(self.X[:,j], residual) - self.reg) / self.M
        
        gradB_0 = -2 * np.sum(self.y - yhat) / self.M

        self.B = self.B - self.lr*gradB
        self.B_0 = self.B_0 - self.lr*gradB_0

        if status:
            print(f"B0: self.B_0 \nB: {self.B}")
        return self


    def predict(self, X):
        """Function to predict values of the target variable given features X and LASSO parameters.

        Parameters
        ----------
        X : numpy.ndarray : feature data with which to predict
        """

        return(X.dot(self.B) + self.B_0)

