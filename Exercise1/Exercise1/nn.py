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
        loops = True
        
        if loops == True:
            
            # 1.1 Task:
            # Compute the class scores for the input.
            # Use the weights and biases to perform a forward propagation and store the results
            # in the scores variable, which should be an array of shape (N, C).
            # Start with a naive implementation with at least 2 loops.
            
            # get number of classes
            C = b2.size
            scores = np.array([N, C])
            
            # create output after first weights
            x1 = np.zeros([W1.shape[1], N])           
            for i in range(N):
                x1[:, i] = W1.T.dot(X[i].T) + b1
            
            # apply ReLU
            # @All: don't know if that is correct, as ReLU is not mention in the task 1.1,
            #       but in the description, at the very top
            ReLU = lambda x: np.max([0,x])
            vectorized_ReLU = np.vectorize(ReLU)
            x1 = vectorized_ReLU(x1)
            
            #calculate network output (scores)
            x2 = np.zeros([W2.shape[1], N])
            for i in range(N):
                x2[:, i] = W2.T.dot(x1[:, i]) + b2
            
            scores = x2.T
            
        else:
            
            # Task 1.2:
            # Now implement the same forward propagation as you did above using no loops.
            # If you are done set the parameter loops to False to test your code. 
        
        
            # @All: Not quite sure if the solution of Task 1.1 should go into here, cause I don#t know how to
            #       do it without any loops, as we have 5 samples.
            ######################################## START OF YOUR CODE ########################################

            pass  # to be replaced by your code
        
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

        pass  # to be replaced by your code
        
        ######################################## END OF YOUR CODE ##########################################

        #--------------------------------------- back propagation -------------------------------------------
        
        grads = {}

        # Task 3: 
        # Compute the derivatives of the weights and biases (back propagation).
        # Store the results in the grads dictionary, where 'W1' referes to the gradient of W1 etc.
        
        ######################################## START OF YOUR CODE ########################################

        pass  # to be replaced by your code

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