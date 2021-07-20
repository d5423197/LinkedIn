import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def neg_sqaured_euclidean_distance(X):
    """
    Input: matrix of X (n X d) (n: the number of samle, d: number of the feature)
    Output: distance matrix D (n X n) 
    D[i ,j] -> neg_sqaured_euclidean_distance of row i (X_i) and row j (X_j) in X
    Reference: https://stackoverflow.com/a/37040451/11972188 (Dont understand the math?)
    """
    r = np.sum(X**2, 1).reshape(-1, 1)
    D = r - 2 * X.dot(X.T) + r.T
    return -D
	
def distance_matrix_by_sigma_square(D, sigma):
    """
    D: distance matrix
    sigma: sigma will be a n length vector. sigma[i] is the variance for the ith row
    """
    # we want sigma to be column vector
    sigma = np.array(sigma).reshape(-1,1) 
    two_sigma_square = 2 * sigma ** 2
    return D / two_sigma_square

def softmax(X, zero_index):
    """
    Compute softmax values for each row of matrix X
    """
    # Normalize the data
    e_x = np.exp(X - np.max(X, axis = 1).reshape([-1, 1]))
    # We want the diagonal to be 0
    if zero_index == None:
        np.fill_diagonal(e_x, 0.)
    else:
        e_x[:, zero_index] = 0.
    # Add a constant to keep numerical stability for log softmax
    c = 1e-5
    e_x = e_x + c
    return e_x / e_x.sum(axis = 1).reshape([-1, 1])

def p_prob_matrix(D, sigma, zero_index):
    '''
    D: distance matrix
    sigma: sigma will be a n length vector. sigma[i] is the variance for the ith row
    ---
    return: probability matrix for p (high-dimensional) p[i,j] => Pj|i
    '''
    # step 2
    D_two_sigma_square = distance_matrix_by_sigma_square(D, sigma)
    # step 3
    return softmax(D_two_sigma_square, zero_index = zero_index)

def perplexity_calc(p):
    """
    p: probability matrix for p
    """
    H_p = -np.sum(p * np.log2(p), axis = 1) # From the equation, apply sum over j which is the column wise direction
    return 2**H_p

def binary_search(row, target, zero_index, left = 1e-20, right = 1000., max_i = 10000, tol = 1e-10):
    """
    row: ith row of distance matrix
    left: lower bound of the range
    right: upper bound of the range
    max_i: maximum iteration
    tol: if val smaller or equal to tol, we will stop
    ---
    return: the optimal sigma for this row
    """
    for i in range(max_i):
        mid = (right + left) / 2.
        # the corresponding probability matrix
        prob_matrix_mid = p_prob_matrix(row, mid, zero_index)
        # the corresponding perplexity val
        perp_val = perplexity_calc(prob_matrix_mid)
        if perp_val > target:
            right = mid
        else:
            left = mid
        if np.abs(perp_val - target) <= tol:
            break
    return mid
	
def find_sigma(distance_matrix, target_perplexity):
    """
    distance_matrix: n by n matrix which we got from step 1
    target_perplexity: parameter which is defined by user
    ---
    return: an array of optimal sigma for each row, sigma[i] -> optimal sigma for ith row
    """
    sigma = []
    for i in range(distance_matrix.shape[0]):
        # here we find optimal sigma for current row
        optimal_sigma_i = binary_search(distance_matrix[i:i+1, :], target_perplexity, i)
        sigma.append(optimal_sigma_i)
    return np.array(sigma)

def x_to_p(X, perplexity):
    """
    X: matrix of X (n X d) (n: the number of samle, d: number of the feature)
    ---
    return: joint probability matrix
    """
    # step 1
    distance = neg_sqaured_euclidean_distance(X)
    # step 7
    sigma = find_sigma(distance, perplexity)
    # step 4
    p = p_prob_matrix(distance, sigma, zero_index = None)
    # step 8
    return (p + p.T) / (2. * p.shape[0])

def q_joint(Y):
    """
    Y: low dimensional representations
    ---
    return: joint probabilities for lower dimension q_ij (n by n matrix)
    """
    # here we use non negative squared euclidean distance
    numerator = 1 / (1 + (-neg_sqaured_euclidean_distance(Y)))
    # diagonal to be zero
    np.fill_diagonal(numerator, 0.)
    # sum over the entire matrix
    denominator = np.sum(numerator)
    return numerator / denominator, numerator

def gradient(p, q, y, q_numerator):
    """
    p: joint probability matrix p
    q: joint probability matrix q
    y: lower dimensional representations
    y_numerator: the distance matrix from step 9
    """
    # you just inset a 'None' at the axis you want to add
    pq_diff = (p - q)[:, :, None] # n*n*1
    y_diff = y[:, None, :] - y[None,:, :] # n*n*2
    q_num_expand = q_numerator[:, :, None] # n*n*1
    grad = np.sum(4. * pq_diff * y_diff * q_num_expand, 1) # n*2
    return grad

def train(X, y, p, num_iterations, learning_rate):
    Y = np.random.normal(0., 0.0001, [X.shape[0], 2])
    P = np.maximum(p, 1e-12)
    for i in range(num_iterations):
        q, q_num = q_joint(Y)
        q = np.maximum(q, 1e-12)
        if (i + 1) % 50 == 0:
            cost = np.sum(p * np.log(p/q))
            print(f'Iteration {i}: error is {cost}')
        grad = gradient(p, q, Y, q_num)
        Y = Y - learning_rate * grad 
    return Y

# global variable
perplexity = 20
num_iters = 1000
learning_rate = 50.

def main(X, y, perplexity, num_iters, learning_rate):
    p = x_to_p(X, perplexity)
    X_low = train(X, y, p, num_iters, learning_rate)
    sns.scatterplot(x = X_low[:, 0], y = X_low[:, 1], hue = y)
    plt.show()
iris = sns.load_dataset('iris')
X = np.array(iris.iloc[:, :-1])
y = np.array(iris.iloc[:, -1])
main(X, y, perplexity, num_iters, learning_rate)