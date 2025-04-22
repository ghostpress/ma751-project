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
        pass


    def fit(self, cross_val=False, nfolds=10, param_vals=None, lr=0.01, iterations=10000):
        """Function to fit LASSO model, using gradient descent on the intercept and feature parameters.
        
        Parameters
        ------
        cross_val : bool : whether to perform cross-validation to tune hyperparameters
        nfolds : int : number of folds for cross-validation
        param_vals : list : candidate regularization term values (if doing cross-validation)
        lr : float : learning rate 
        iterations : int : number of gradient descent iterations
        """

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \nFitting model.\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f"Initial model: \nB_0: {self.B_0} \nB: {self.B} \nlambda: {self.reg}")

        # Cross-validation to choose the best regularization value
        if cross_val:
            print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \nPerforming {nfolds}-fold cross-validation... \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            best_lambda = self.cross_validation(nfolds, param_vals, lr=lr, iterations=iterations)
            self.reg = best_lambda

            # Now update linear LASSO parameters
            for k in range(iterations):
                self.gradient_descent_update(lr=lr)
                    

        else:  # regular gradient descent on full training set
            for k in range(iterations):
                if k % 1000 == 0:
                    print(f"Iteration: {k}/{iterations}")  # give a status update every (niter/1000) iterations
                    self.gradient_descent_update(status=True)

        print(f"Updated model: \nB_0: {self.B_0} \nB: {self.B} \nlambda: {self.reg}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
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

                curr_model = LassoRegression(train_X, train_y, p)

                for k in range(iterations):
                    curr_model.gradient_descent_update(lr=lr) 
                
                yhat_val = curr_model.predict(val_X)
                err = self.MSE(yhat_val, val_y)
                overall_err += err
                #print(f"Fold {fold} MSE: {round(err, 5)} with parameter value {p}.")
            
            overall_err = overall_err / nfolds
            errors[p] = overall_err
            print(f"Overall MSE: {round(overall_err, 7)} with parameter value {p}.")

        best_param = min(errors, key=errors.get)
        self.reg = best_param
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \nAfter {nfolds}-fold cross validation, best lambda value is {best_param}. \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        return best_param


    def MSE(self, pred, truth):
        """Function to calculate MSE between model prediction and a target value.

        Parameters
        ----------
        pred : np.ndarray : predicted output
        truth : np.ndarray : target output
        """

        residual = np.subtract(pred, truth)
        mse = (residual**2).mean()
        return mse


    def gradient_descent_update(self, lr=0.01, status=False):
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

        self.B = self.B - lr*gradB
        self.B_0 = self.B_0 - lr*gradB_0

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

