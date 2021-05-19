import numpy as np

def compute_cost(X, y, theta, m):
    """
    Compute the cost of a particular choice of theta for linear regression.

    Input Parameters
    ----------------
    X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
        m= number of training examples
        n= number of features (including X_0 column of ones)
    y : 1D array of labels/target value for each traing example. dimension(1 x m)

    theta : 1D array of fitting parameters or weights. Dimension (1 x n)

    Output Parameters
    -----------------
    J : Scalar value.
    """
    predictions = X.dot(theta)
      #print('predictions= ', predictions[:5])
    errors = np.subtract(predictions, y)
      #print('errors= ', errors[:5]) 
    sqrErrors = np.square(errors)
      #print('sqrErrors= ', sqrErrors[:5]) 
      #J = 1 / (2 * m) * np.sum(sqrErrors)
      # OR
      # We can merge 'square' and 'sum' into one by taking the transpose of matrix 'errors' and taking dot product with itself
      # If your confuse about this try to do this with few values for better understanding  
    J = 1/(2 * m) * errors.T.dot(errors)

    return J

def gradient_descent(X, y, theta, alpha, iterations, m):
    """
    Compute cost for linear regression.
  
    Input Parameters
    ----------------
    X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
        m= number of training examples
        n= number of features (including X_0 column of ones)
    y : 1D array of labels/target value for each traing example. dimension(m x 1)
    theta : 1D array of fitting parameters or weights. Dimension (1 x n)
    alpha : Learning rate. Scalar value
    iterations: No of iterations. Scalar value. 

    Output Parameters
    -----------------
    theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
    cost_history: Conatins value of cost for each iteration. 1D array. Dimansion(m x 1)
    """
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        #print('predictions= ', predictions[:5])
        errors = np.subtract(predictions, y)
        #print('errors= ', errors[:5])
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        #print('sum_delta= ', sum_delta[:5])
        theta = theta - sum_delta;

    cost_history[i] = compute_cost(X, y, theta, m)  
  
    return theta, cost_history

def mlr_predict(X_test, test_theta):
    prediction = list()
    for i in range(len(X_test)):
        test_data = np.hstack((np.ones(1), X_test[i,]))
        prediction.append(test_data.dot(test_theta))

    return prediction