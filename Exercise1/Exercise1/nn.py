import numpy as np
import time

class TwoLayerNet(object):
    """
    A two-layer neural network with the architecture:
    
    input - fully connected layer - activation function (ReLU) - fully connected layer - softmax 
    
    The neural network performes a classification over C classes. The output is a score of the classes.
    A softmax loss function and a L2 regularization are used when training the network.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        """
        Inputs:
        
        - input_dim: Dimension of the input
        - hidden_dim: Neurons in hidden layer
        - output_dim: Classes
        
        Variables (stored in a dictionary self.params):
        
        W1: First layer weights with shape (L, M)
        b1: First layer biases with shape (M,)
        W2: Second layer weights with shape (M, N)
        b2: Second layer biases with shape (N,)
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = std * np.random.randn(hidden_dim, output_dim)
        self.params['b2'] = np.zeros(output_dim)

    def loss_grad(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients.

        Inputs:
        
        X: Input data of shape (N, D). Each X[i] is a training sample.
        y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
           an integer in the range 0 <= y[i] < C. This parameter is optional; if it
           is not passed then we only return scores, and if it is passed then we
           instead return the loss and gradients.
        reg: Regularization strength.

        Returns:
        
        - If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
          the score for class c on input X[i].
        - If y is not None, return a tuple of:
          - loss: Loss (data loss and regularization loss) for this batch of training
            samples.
          - grads: Dictionary mapping parameter names to gradients of those parameters
            with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2'] 
        b2 = self.params['b2']
        N, D = X.shape

        #--------------------------------------- forward propagation ---------------------------------------                     
        
        scores = None
        loops = False
        
        if loops == True:
            
            # 1.1 Task:
            # Compute the class scores for the input.
            # Use the weights and biases to perform a forward propagation and store the results
            # in the scores variable, which should be an array of shape (N, C).
            # Start with a naive implementation with at least 2 loops.

            ######################################## START OF YOUR CODE ########################################
            # start = time.time()
            # calculate dimensions
            M, C = W2.shape
            # fully connected layer
            z1 = np.zeros([N, M])
            for i in range(N):
                for j in range(M):
                    for k in range(D):
                        z1[i][j] += X[i][k] * W1[k][j]
                    # add bias
                    z1[i][j] += b1[j]
            # ReLU
            a1 = np.maximum(0, z1)
            # second fully connected layer
            z2 = np.zeros([N, C])
            for i in range(N):
                for j in range(C):
                    for k in range(M):
                        z2[i][j] += a1[i][k] * W2[k, j]
                    # add bias
                    z2[i][j] += b2[j]
            end = time.time()
            # print('time with loops: ')
            # print(end - start)
            scores = z2
            ######################################## END OF YOUR CODE ##########################################

        else:
            
            # Task 1.2:
            # Now implement the same forward propagation as you did above using no loops.
            # If you are done set the parameter loops to False to test your code.

            ######################################## START OF YOUR CODE ########################################
            # start = time.time()
            z1 = np.dot(X, W1) + b1
            a1 = np.maximum(0, z1)
            z2 = np.dot(a1, W2) + b2
            end = time.time()
            # print('time without loop: ')
            # print(end - start)
            scores = z2
        ######################################## END OF YOUR CODE ##########################################

        # Jump out if y is not given.
        if y is None:
            return scores

        #--------------------------------------- loss function ---------------------------------------------
        
        loss = None
        
        # Task 2:
        # Compute the loss with softmax and store it in the variable loss. Include L2 regularization for W1 and W2.
        # Make sure to handle numerical instabilities.

        ######################################## START OF YOUR CODE ########################################

        def softmax(x):
            res = np.exp(x - np.max(x))
            return res / np.sum(res)

        a2 = np.zeros(z2.shape)
        for i in range(z2.shape[0]):
            a2[i] = softmax(z2[i])

        loss = 1 / N * np.sum(-np.log(a2[range(N), y])) + reg * np.sum(np.square(W1)) + reg * np.sum(np.square(W2))

        ######################################## END OF YOUR CODE ##########################################

        #--------------------------------------- back propagation -------------------------------------------
        
        grads = {}

        # Task 3:
        # Compute the derivatives of the weights and biases (back propagation).
        # Store the results in the grads dictionary, where 'W1' referes to the gradient of W1 etc.
        
        ######################################## START OF YOUR CODE ########################################

        dL_dz2 = a2
        dL_dz2[range(N), y] -= 1

        dz2_dw2 = a1

        dL_dw2 = 1 / N * np.dot(dz2_dw2.T, dL_dz2)

        dL_db2 = np.zeros(b2.shape[0])
        for i in range(b2.shape[0]):
            dL_db2[i] = np.mean(dL_dz2[range(N), i])

        def reluDerivative(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

        dL_dz1 = np.dot(W2, dL_dz2.T) * reluDerivative(z1.T)

        dL_dw1 = 1 / N * np.dot(dL_dz1, X)
        dL_dw1 = dL_dw1.T

        dL_db1 = np.zeros(b1.shape[0])
        for i in range(b1.shape[0]):
            dL_db1[i] = np.mean(dL_dz1.T[range(N), i])

        grads = {'W1': dL_dw1, 'b1': dL_db1, 'W2': dL_dw2, 'b2': dL_db2}

        ######################################## END OF YOUR CODE ##########################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            # Task 4.1: 
            # Create a random minibatch of training data X and labels y, and stor
            # them in X_batch and y_batch.
            ######################################## START OF YOUR CODE ########################################

            pass  # to be replaced by your code

            ######################################## END OF YOUR CODE ##########################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss_grad(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # Task 4.2:
            # Update the parameters of the network (in self.params) by using stochastic gradient descent. 
            # You will need to use the gradients in the grads dictionary.
            ######################################## START OF YOUR CODE ########################################

            pass  # to be replaced by your code

            ######################################## END OF YOUR CODE ##########################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained network to predict labels for the data points. 
        For each data point we predict scores for each of the C classes, 
        and assign each data point to the class with the highest score.

        Inputs:
        
        X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        
        y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c.
        """
        y_pred = None

        # Task 4.3: 
        # Implement this function to predict labels for the data points.
        ######################################## START OF YOUR CODE ########################################

        pass  # to be replaced by your code

        ######################################## END OF YOUR CODE ##########################################

        return y_pred