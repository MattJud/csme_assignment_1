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
        hidden_dim, output_dim = W2.shape
        if loops == True:
            
            # 1.1 Task:
            # Compute the class scores for the input.
            # Use the weights and biases to perform a forward propagation and store the results
            # in the scores variable, which should be an array of shape (N, C).
            # Start with a naive implementation with at least 2 loops.
        
            ######################################## START OF YOUR CODE ########################################

            
            z1 = np.zeros([N, hidden_dim])
            z2 = np.zeros([N, output_dim])
            
            # toDo: to be completed
            
            #pass #to be replaced by your code
        
            ######################################## END OF YOUR CODE ##########################################
            
        else:
            
            # Task 1.2:
            # Now implement the same forward propagation as you did above using no loops.
            # If you are done set the parameter loops to False to test your code. 
        
            ######################################## START OF YOUR CODE ########################################

            
            b1 = b1.reshape(b1.shape[0],1) # make sure that it is a matrix of shape [xx,yy] and not [xx,]
            z1 = np.dot(X, W1) + b1.T
            a1 = np.maximum(0,z1) #Relu function
            
            b2 = b2.reshape(b2.shape[0],1)
            z2 = np.dot(a1,W2) + b2.T
            #expo = np.exp(z2)
            #expo_sum = np.sum(np.exp(z2),axis=1 , keepdims=True)
            #a2 = expo/expo_sum #softmax function
            #print('expo.shape = ' , expo.shape)
            #print('expo_sum.shape = ' , expo_sum.shape)
            scores = z2
            #print(a2)
            #for i in range(N):
                
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
        # function to calculate a stable softmax of a given array (handle numerical instabilities)
        def softmax_stable(x):
            e_res = np.exp(x - np.max(x))
            return e_res / np.sum(e_res)
            
        # Calculate softmax
        expo = np.exp(z2)
        expo_sum = np.sum(np.exp(z2),axis=1 , keepdims=True)
        prob = expo/expo_sum
        
        prob_stable = np.zeros([N, b2.shape[0]])
        for i in range(N):
            prob_stable[i] = softmax_stable(z2[i])
            
        # Calculate loss
        
        #print("y.shape", y.shape)
        #print("y = ", y)
        #print("prob.shape", prob.shape)
        log_likelihood = -np.log(prob[range(N),y])
        #print(prob)
        #print(log_likelihood)
        
        loss = np.sum(log_likelihood) / N + reg * (sum(sum(np.square(W1)))+sum(sum(np.square(W2))))
        
        ######################################## END OF YOUR CODE ##########################################

        #--------------------------------------- back propagation -------------------------------------------
        
        grads = {}

        # Task 3: 
        # Compute the derivatives of the weights and biases (back propagation).
        # Store the results in the grads dictionary, where 'W1' referes to the gradient of W1 etc.
        
        ######################################## START OF YOUR CODE ########################################

        # calculate the gradient of loss regatrding to softmax cross entropy
        # http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
        dz2 = prob
        dz2[range(N),y] -= 1
        #dz2 = 1/N * dz2
        #print(dz2.shape)
        
        # to be deleted later
        #print("log_likelihood.shape = ",log_likelihood.shape)
        #dz2 = log_likelihood.reshape(log_likelihood.shape[0],1) # A2 - Y #error
        #dz2 = np.sum(log_likelihood) / N #
        #dz2 = prob - y.reshape
        
        # Notation: see Andrew Ng
        # https://github.com/Kulbear/deep-learning-coursera/blob/master/Neural%20Networks%20and%20Deep%20Learning/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb
        dW2 = 1/N * np.dot(a1.T, dz2)
        db2 = 1/N * np.sum(dz2, axis=0, keepdims=True).T
        #print("dz2.shape = ",dz2.shape)
        #print("a1.shape = ", a1.shape)
        #print("b2.shape = ",b2.shape)
        #print("db2.shape = ",db2.shape)
        #print("W2.shape = ", W2.shape)
        #print("dW2.shape = " , dW2.shape)  


        dz1 = np.dot(W2, dz2.T) # should still be multiplied by relu_gradient
        #print("dz1.shape = " , dz1.shape)   
        
        # calculate relu gradient
        dz1_resh = dz1.reshape(dz1.shape[0]*dz1.shape[1],1)
        z1_resh = z1.T.reshape(z1.T.shape[0]*z1.T.shape[1],1)
 
        #print("dz1_resh.shape = " , dz1_resh.shape)   
        #print("z1_resh.shape = " , z1_resh.shape)
        
        dz1_resh[z1_resh < 0] = 0  # the derivetive of ReLU
        dz1 = dz1_resh.reshape(dz1.shape[0], dz1.shape[1]) 
        dz1 = dz1.T
        #print("dz1.shape = " , dz1.shape) 
        
        dW1 = 1/N * np.dot(X.T,dz1)
        db1 = 1/N * np.sum(dz1, axis=0, keepdims=True).T
        #print("X.shape = ", X.shape)
        #print("dW1.shape = ", dW1.shape)
        #print("b1.shape = ",b1.shape)
        #print("db1.shape = ",db1.shape)
        
        grads = {"W1": dW1,
             "b1": db1,
             "W2": dW2,
             "b2": db2}
        #pass  # to be replaced by your code

        ######################################## END OF YOUR CODE ##########################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=5, verbose=False):
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
            
            arr = np.arange(batch_size)
            arr = np.random.permutation(arr)
            print(X.shape[0])
            X_batch = X[arr][:]
            y_batch = y[arr]
            
            
            
            pass  # to be replaced by your code

            ######################################## END OF YOUR CODE ##########################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss_grad(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # Task 4.2:
            # Update the parameters of the network (in self.params) by using stochastic gradient descent. 
            # You will need to use the gradients in the grads dictionary.
            ######################################## START OF YOUR CODE ########################################

            # steps:
            # 1: Forward propagation --> cost and Loss     
            # 2: Backpropagation. --> grad
            # 3: Gradient descent parameter update
            
            # extract parameters, update them and store them again
            dW1 = grads['W1']
            db1 = grads['b1']
            dW2 = grads['W2']
            db2 = grads['b2']
            
            W1 = self.params['W1']
            b1 = self.params['b1']
            W2 = self.params['W2'] 
            b2 = self.params['b2']
            
            b1 = b1.reshape(b1.shape[0],1) # make sure that it is a matrix of shape [xx,yy] and not [xx,]
            b2 = b2.reshape(b2.shape[0],1)
            
            self.params['W1'] = W1 - learning_rate * dW1
            self.params['b1'] = b1 - learning_rate * db1
            self.params['W2'] = W2 - learning_rate * dW2
            self.params['b2'] = b2 - learning_rate * db2
            

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

        scores = self.loss_grad(X)
        y_pred = np.argmax(scores, axis=1)
        pass  # to be replaced by your code

        ######################################## END OF YOUR CODE ##########################################

        return y_pred