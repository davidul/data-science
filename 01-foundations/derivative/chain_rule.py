import numpy as np
from typing import Callable, Dict, Tuple, List

Array_Function = Callable[[np.ndarray], np.ndarray] # [input], output
Chain = List[Array_Function]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def square(x: np.ndarray) -> np.ndarray:
    return np.power(x, 2)

def deriv(func: Callable[[np.ndarray], np.ndarray],
          input_: np.ndarray,
          delta: float = 0.001) -> np.ndarray:
    '''
    Evaluates the derivative of a function "func" at every element in the
    "input_" array.
    See https://en.wikipedia.org/wiki/Numerical_differentiation
    '''
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

def chain_length_2(chain: Chain,
                     x: np.ndarray) -> np.ndarray:
    assert len(chain) == 2, \
        "Length of input 'chain' should be 2"
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(x))

def chain_length_3(chain: Chain,
                     x: np.ndarray) -> np.ndarray:
    assert len(chain) == 3, \
        "Length of input 'chain' should be 3"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]
    return f3(f2(f1(x)))

def chain_derivative(chain: Chain, input_range: np.ndarray) -> np.ndarray:
    '''
    Uses the chain rule to compute the derivative of two nested functions:
    (f2(f1(x)))' = f2'(f1(x)) * f1'(x)

    chain[0] = f1
    chain[1] = f2

    Args:
        chain (Chain): List of two functions (with one input and one output)
        input_range (np.ndarray): ndarray of numbers (input data)
    '''
    assert len(chain) == 2, \
        "This function requires 'Chain' objects of length 2"

    assert input_range.ndim == 1, \
        "Function requires a 1 dimensional ndarray as input_range"
    
    f1 = chain[0]
    f2 = chain[1]
    # df1/dx
    f1_of_x = f1(input_range)
    # df1/du
    df1dx = deriv(f1, input_range)

    # df2/du(f1(x))
    df2du = deriv(f2, f1(input_range))
    return df1dx * df2du

def chain_deriv_3(chain: Chain,
                  input_range: np.ndarray) -> np.ndarray:
    '''
    Uses the chain rule to compute the derivative of three nested functions:
    (f3(f2(f1)))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    '''

    assert len(chain) == 3, \
    "This function requires 'Chain' objects to have length 3"

    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_of_x = f1(input_range)

    # f2(f1(x))
    f2_of_x = f2(f1_of_x)

    # df3du
    df3du = deriv(f3, f2_of_x)

    # df2du
    df2du = deriv(f2, f1_of_x)

    # df1dx”

    df1dx = deriv(f1, input_range)

    # Multiplying these quantities together at each point
    return df1dx * df2du * df3du

def manual_forward(X: np.ndarray,W: np.ndarray) -> np.ndarray:
    assert X.shape[1] == W.shape[0], \
        '''
        For matrix multiplication, the number of columns in the first array should
        match the number of rows in the second, instead the number of columns in X is {}
        and the number of rows in W is {}
        '''.format(X.shape[1], W.shape[0])
    N = np.dot(X,W)
    return N

def matrix_function_backward_1(X: np.ndarray,
                               W: np.ndarray,
                               sigma: Array_Function) -> np.ndarray:
    '''
    Computes the derivative of our matrix function with respect to
    the first element.
    '''
    assert X.shape[1] == W.shape[0]

    # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # backward calculation
    dSdN = deriv(sigma, N)

    # dNdX
    dNdX = np.transpose(W, (1, 0))

    # multiply them together; since dNdX is 1x1 here, order doesn't matter
    return np.dot(dSdN, dNdX)

def matrix_function_forward_sum(X: np.ndarray, W: np.ndarray,
                                    sigma: Array_Function) -> float:
    '''
    Computing the result of the forward pass of this function with input ndarrays X and W and function sigma.
    '''
    assert X.shape[1] == W.shape[0]
            # matrix multiplication
    N = np.dot(X, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)
    # sum all the elements
    L = np.sum(S)
    return L

def forward_linear_regression(X_batch: np.ndarray, y_batch: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:
    '''
    Forward pass for the step-by-step linear regression.
    '''
    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights['W'].shape[0]
    assert weights['B'].shape[0] == weights['W'].shape[1]

    # compute the operations on the forward pass
    N = np.dot(X_batch, weights['W'])
    P = N + weights['B']
    L = np.mean(np.power(y_batch - P, 2))

    # save the activations; you'll need them for the backward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P

    return L, forward_info

def loss_gradients(forward_info: Dict[str, np.ndarray], weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    batch_size = forward_info['X'].shape[0]

    dLdP = -2 * (forward_info['X'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B'])
    dLdN = dLdP * dPdN
    dnDw = np.transpose(forward_info['X'], (1, 0))
    dLDw = np.dot(dnDw, dLdN)
    dLdB = (dLdP * dPdB).sum(axis=0)

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB

    return loss_gradients


def leaky_relu(X: np.ndarray) -> np.ndarray:
    '''
    Computes Leaky ReLU activation function
    '''
    return np.maximum(0.2 * X, X)

def multiple_inputs_add(x: np.ndarray, y: np.ndarray, sigma: Array_Function) -> float:
    '''
    Function with multiple inputs and addition
    '''
    assert x.shape == y.shape

    a = x + y
    return sigma(a)

def multiple_inputs_add_backward(x: np.ndarray, y: np.ndarray, sigma: Array_Function) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the derivative of this function with respect to its inputs.
    '''
    # Compute the forward pass
    a = x + y
    # Compute the derivative with respect to a
    dsda = deriv(sigma, a)
    # Compute the derivative with respect to x, y, and a
    dadx = 1.0 * dsda
    dady = 1.0 * dsda
    return dsda, dadx, dady

def matmul_forward(X: np.ndarray,
                   W: np.ndarray) -> np.ndarray:
    '''
    Computes the forward pass of a matrix multiplication.
    '''

    assert X.shape[1] == W.shape[0], \
    '''
    For matrix multiplication, the number of columns in the first array should
    match the number of rows in the second; instead the number of columns in the
    first array is {0} and the number of rows in the second array is {1}.
    '''.format(X.shape[1], W.shape[0])

    # matrix multiplication
    N = np.dot(X, W)

    return N