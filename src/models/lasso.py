import numpy as np

class LassoRegression():
    """Class to implement LASSO regression, based on (3.52) in Hastie's 'Elements of Statistical Learning' (Springer, 2013).
    The model is trained using gradient descent. 
    """

    def __init__(self, X, y, reg):
        """
        Parameters
        ------
        X : numpy.ndarray : column data
        y : numpy.ndrarry : target data
        reg : float : regularization term
        """

        self.M, self.N = X.shape
        self.B = np.zeros(self.N)
        self.B_0 = 0
        self.X = np.array(X)
        self.y = np.array(y)
        self.reg = reg

        self.cross_validation()
        pass


    def fit(self, lr=0.01, iterations=10000):
        """Function to fit LASSO model, using gradient descent on the intercept and feature parameters.
        
        Parameters
        ------
        X : numpy.ndarray : column data
        y : numpy.ndrarry : target data
        reg : float : regularization term
        lr : float : learning rate 
        """

        self.lr = lr
        self.iterations = iterations

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


    def _kfolds(self, k):
        """Helper function to create k-fold training and validation sets, using the indices of the full training set.

        Parameters
        ----------
        k : int : number of folds
        """

        fold_size = self.X.shape[0] // k 
        folds = {}

        # For each fold...
        for i in range(k):
            all_indices = np.arange(self.X.shape[0])
            test_indices = np.random.choice(all_indices, size=fold_size, replace=False)   # randomly select the test indices
            train_indices = [int(idx) for idx in all_indices if idx not in test_indices]  # use all other indices for training

            folds[i] = train_indices, test_indices

        return folds


    def cross_validation(self, k=10):
        fold_indices = self._kfolds(k)
        pass


    def gradient_descent_update(self, status=False):
        """Function to perform gradient-descent updates on the LASSO parameters. 

        Parameters
        ----------
        status : bool : whether to print a status update on the parameter values and MSE in a given iteration
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

