import numpy as np

class RidgeRegression():
    """Class to implement ridge regression, based on (3.42) in Hastie's 'Elements of Statistical Learning' (Springer, 2013).
    The model is trained using gradient descent. 
    """

    def __init__(self, X, y, reg, B):
        """
        Parameters
        ------
        X : numpy.ndarray : column data
        y : numpy.ndrarry : target data
        reg : float : regularization term
        B_0 : float : intercept term
        B : numpy.ndarray : linear parameters
        """

        self.M, self.N = X.shape
        self.X = np.array(X, dtype="int64")
        #self.X = np.delete(self.X, 3, 1)
        self.y = np.array(y, dtype="int64")
        self.reg = reg
        #self.B_0 = B_0 
        self.B = B

        #print(self.X.shape)
        #print(self.B.shape)
        self.errors = {} # field to track error over training iterations
        pass


    @classmethod
    def from_params(cls, X, y, reg, B):
        return cls(X, y, reg, B)


    @classmethod
    def new_model(cls, X, y, reg=0.0):
        # Initialize model parameter values
        M, N = X.shape
        B = np.random.randn(N) #np.zeros(N)
        #B_0 = 0.0
        return cls(X, y, reg, B)


    def fit(self, cross_val=False, nfolds=10, param_vals=None, lr=0.001, iterations=10000):
        """Function to fit ridge model, using gradient descent on the intercept and feature parameters.
        
        Parameters
        ------
        cross_val : bool : whether to perform cross-validation to tune hyperparameters
        nfolds : int : number of folds for cross-validation
        param_vals : list : candidate regularization term values (if doing cross-validation)
        lr : float : learning rate 
        iterations : int : number of gradient descent iterations
        """

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nFitting model.\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Initial model: \nB: {self.B} \nlambda: {self.reg}")

        # Cross-validation to choose the best regularization value
        if cross_val:
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \nPerforming {nfolds}-fold cross-validation... \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            best_lambda = self.cross_validation(nfolds, param_vals, lr=lr, iterations=iterations)
            self.reg = best_lambda

            # Now update linear ridge parameters
            for k in range(iterations):
                self.gradient_descent_update(lr=lr)
                    
        # Regular gradient descent on full training set
        else: 
            for k in range(iterations):
                if k % 1000 == 0:
                    print(f"Iteration: {k}/{iterations}")  # give a status update every (niter/1000) iterations
                    self.gradient_descent_update(status=True)

        print(f"\nUpdated model: \nB: {self.B} \nlambda: {self.reg}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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


    def cross_validation(self, nfolds, param_vals, lr=0.01, iterations=10000):
        """Function to perform k-fold cross-validation for selecting the best hyperparameter (in this case, regularizer lambda) value. 

        Parameters
        ----------
        nfolds : int : number of folds
        param_vals : list : potential parameter values from which to select the best one
        lr : float : gradient descent learning rate
        iterations : int : number of gradient descent iterations
        """

        folds = self._kfolds(nfolds)
        errors = {}

        for p in param_vals:
            overall_err = 0

            for fold in folds.keys():
                train_ind = folds[fold][0]
                val_ind = folds[fold][1]

                train_X = self.X[train_ind,:]
                train_y = self.y[train_ind]
                val_X = self.X[val_ind,:]
                val_y = self.y[val_ind]

                curr_model = RidgeRegression.new_model(train_X, train_y, p)

                for k in range(iterations):
                    curr_model.gradient_descent_update(lr=lr) 
                
                yhat_val = curr_model.predict(val_X)
                err = self.error(yhat_val, val_y)
                overall_err += err
                #print(f"Fold {fold} MSE: {round(err, 5)} with parameter value {p}.")
            
            overall_err = overall_err / nfolds
            errors[p] = overall_err
            print(f"Overall error: {round(overall_err, 7)} with parameter value {p}.")

        best_param = min(errors, key=errors.get)
        self.reg = best_param
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \nAfter {nfolds}-fold cross validation, best lambda value is {best_param}. \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        return best_param


    def error(self, pred, truth, func="abs"):
        """Function to calculate error between model prediction and a target value.

        Parameters
        ----------
        pred : np.ndarray : predicted output
        truth : np.ndarray : target output
        """

        residual = np.subtract(pred, truth)

        if func == "abs":
            err = np.abs(residual).mean()
        elif func == "mse":
            err = (residual**2).mean()
        return err


    def gradient_descent_update(self, lr=0.01, status=False):
        """Function to perform gradient-descent updates on the ridge parameters. 

        Parameters
        ----------
        status : bool : whether to print a status update on the parameter values and error in a given iteration
        """

        yhat = self.predict(self.X)
        gradB = np.zeros(self.N)
        residual = np.subtract(self.y, yhat)
        
        if status:
            print(f"Error: {(np.square(residual)).mean()}")

        gradB = (2/self.M) * (np.transpose(self.X)).dot(residual) + (2 * self.reg * self.B)  # FIXME: print gradB to debug
        self.B = self.B - lr*gradB
        #self.B_0 = self.B_0 - lr*gradB_0

        if status:
            print(f"B: {self.B}")
        return self


    def predict(self, X):
        """Function to predict values of the target variable given features X and ridge parameters.

        Parameters
        ----------
        X : numpy.ndarray : feature data with which to predict
        """
        return(X.dot(self.B)) #+ self.B_0)


    def solve(self):
        design = np.linalg.inv(np.dot(np.transpose(self.X), self.X) + (self.reg * np.identity(self.N-1)))
        betaHat = np.dot(design, np.dot(np.transpose(self.X), self.y))
        return betaHat


    def save(self, destination):

        np.save(destination + "beta.npy", self.B)
        #np.save(destination + "beta0.npy", self.B_0)
        np.save(destination + "lambda.npy", self.reg)
        return self

