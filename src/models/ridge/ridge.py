import numpy as np
import utils  # TODO: delete

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
        self.y = np.array(y, dtype="int64")
        self.reg = reg
        self.B = B
        pass


    @classmethod
    def from_params(cls, X, y, reg, B):
        return cls(X, y, reg, B)


    @classmethod
    def new_model(cls, X, y, reg=0.00001):
        # Initialize model parameter values
        M, N = X.shape
        B = np.random.randn(N)
        return cls(X, y, reg, B)


    def fit(self, cross_val=False, nfolds=10, param_vals=None):
        """Function to fit ridge model, using gradient descent on the intercept and feature parameters.
        
        Parameters
        ------
        cross_val : bool : whether to perform cross-validation to tune hyperparameters
        nfolds : int : number of folds for cross-validation
        param_vals : list : candidate regularization term values (if doing cross-validation)
        lr : float : learning rate 
        iterations : int : number of gradient descent iterations
        """

        # Cross-validation to choose the best regularization value
        if cross_val:
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \nPerforming {nfolds}-fold cross-validation... \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            best_lambda = self.cross_validation(nfolds, param_vals)
            self.reg = best_lambda
                    
        else: 
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nFitting model.\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            #print(f"Initial model: \nB: {self.B} \nlambda: {self.reg}")
            B = self.solve(self.X, self.y)
            self.B = B


        print(f"\nFitted model: \nB: {self.B} \nlambda: {self.reg}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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


    def cross_validation(self, nfolds, param_vals):
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
        overall_err = {}

        for p in param_vals:
            folds_err = np.zeros(nfolds)

            for fold in folds.keys():
                train_ind = folds[fold][0]
                val_ind = folds[fold][1]

                train_X = self.X[train_ind,:]
                train_y = self.y[train_ind]
                val_X = self.X[val_ind,:]
                val_y = self.y[val_ind]

                curr_model = RidgeRegression.new_model(train_X, train_y, p)
                curr_model.solve(train_X, train_y)
                #print(curr_model.B)
                
                yhat_val = curr_model.predict(val_X)
                err = self.error(yhat_val, val_y)
                folds_err[fold] = err

            errors[p] = folds_err
            print(f"Errors in {nfolds} folds for parameter value {p}: ", errors[p], 5)
            print("Mean error over all folds: ", errors[p].mean(), 5)
            overall_err[p] = errors[p].mean()

        best_param = min(overall_err, key=overall_err.get)
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


    def predict(self, X):
        """Function to predict values of the target variable given features X and ridge parameters.

        Parameters
        ----------
        X : numpy.ndarray : feature data with which to predict
        """
        return(X.dot(self.B))


    def solve(self, X, y):
        """Function to compute the Beta-hat parameters from data X and y, based on the known formula for 
        the minimizer of the squared loss in ridge regression.

        Parameters
        ----------
        X : numpy.ndarray : feature data
        y : numpy.ndarray : target data
        """

        A = self.reg * np.identity(self.N)
        matrix = np.dot(X.T, X) + A

        try: 
            result = np.linalg.inv(matrix).dot((X.T).dot(y))

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                rows_equal = len(np.unique(matrix, axis=0)) < matrix.shape[0]
                cols_equal = len(np.unique(matrix, axis=1)) < matrix.shape[1]
                cols_zero = utils.nonzero(matrix).shape != matrix.shape
                rows_zero = utils.nonzero(matrix.T).shape != (matrix.T).shape
                
                # Check if any rows or columns are equal
                if rows_equal or cols_equal:
                    raise ValueError(f'The matrix to be inverted in the solve step has rows or columns that are equal: rows check returned {rows_equal} and columns check returned {cols_equal}.')

                # Check if any rows or columns have all-zero values
                elif rows_zero or cols_zero:
                    raise ValueError(f'The matrix to be inverted in the solve step has rows or columns that are all zero: rows check returned {rows_zero} and columns check returned {cols_zero}.')

                # For some computational reason, the matrix is still non-singular - add some small noise independent of the response y
                else:
                    print('Adding some noise to the matrix to make inversion possible.')
                    matrix = matrix + (np.random.random(matrix.shape) / 10)  #  noise in range [0, 0.1)
                    result = np.linalg.inv(matrix).dot((X.T).dot(y))
        
        return result


    def save(self, destination):
        """Function to save fitted model parameters to files in a given directory.

        Parameters
        ----------
        destination : str : path to the output directory
        """

        np.save(destination + "beta.npy", self.B)
        np.save(destination + "lambda.npy", self.reg)
        return self

