import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel

def sigmoid(X):
    '''Compute the sigmoid function '''
    #d = zeros(shape=(X.shape))

    den = 1.0 + e ** (-1.0 * X)

    d = 1.0 / den

    return d


def compute_cost(theta,X,y): #computes cost given predicted and actual values
    m = X.shape[0] #number of training examples
    theta = np.reshape(theta,(len(theta),1))

    #y = reshape(y,(len(y),1))

    J = (1./m) * (-np.transpose(y).dot(np.log(sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))

    grad = np.transpose((1./m)*np.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0][0]#,grad


def compute_grad(theta, X, y):

    #print theta.shape

    theta.shape = (1, 3)

    grad = np.zeros(3)

    h = sigmoid(X.dot(theta.T))

    delta = h - y

    l = grad.size

    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / np.m) * sumdelta * - 1

    theta.shape = (3,)

    return  grad

#load the dataset
data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

pos = np.where(y == 1) #indices
neg = np.where(y == 0) #0's indices
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
show()
