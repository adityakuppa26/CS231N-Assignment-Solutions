from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    Z = np.dot(X, W)
    yhat=None
    N, D, C = X.shape[0], W.shape[0], W.shape[1]
    for i in range(N):
      exp_term = np.exp(Z[i] - np.max(Z[i]))
      yhat = (exp_term/np.sum(exp_term))
      loss += -np.log(yhat[y[i]])
      yhat[y[i]] -= 1
      dW += np.dot(X[i].reshape((D, 1)), yhat.reshape((1, C)))
    dW /= N
    loss /= N
    loss += reg*np.sum(np.square(W))
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    Z = np.dot(X, W)  #NxC
    exp_term = np.exp(Z-np.max(Z, axis=1, keepdims=True))
    yhat = exp_term/np.sum(exp_term, axis=1, keepdims=True)

    loss += np.sum(-np.log(yhat[np.arange(N), y]))
    loss /= N
    loss += reg*np.sum(np.square(W))

    dz = np.copy(yhat)
    dz[np.arange(N), y] -= 1
    dz /= N
    dW = np.dot(X.T, dz) #DxN * NxC = DxC
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
