import numpy as np
import time
from tqdm import tqdm


class FISTA(object):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for
    solving Group Lasso problem.
    """

    def __init__(
        self,
        blocks,
        design_matrix,
        response,
        lambda_,
        tolerance=1e-10,
        verbose=False,
        time_crit=None,
    ):
        """
        Initialize the BlockCoordinateDescentGroupLasso object

        Parameters
        ----------
        blocks : list of np.ndarray
            List of indices of the features corresponding to each block
        design_matrix : np.ndarray of shape (num_observations, num_features)
            Design matrix for the dataset
        response : np.ndarray of shape (num_observations, 1)
            Response vector for the dataset
        lambda_ : float
            Regularization parameter for Group Lasso
        tolerance : float
            Tolerance for the optimization algorithm
        verbose : bool
            Verbosity flag
        time_crit : float
            Time constraint for the optimization algorithm
        """

        self.start_time = time.time()
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.design_matrix = design_matrix
        self.response = response
        self.lambda_ = lambda_
        self.num_observations, self.num_features = design_matrix.shape
        self.tolerance = tolerance
        self.verbose = verbose
        self.time_crit = time_crit

        self.A_transpose_A = design_matrix.T @ design_matrix
        self.A_transpose_b = design_matrix.T @ response
        self.smoothness_constant = self.__getSmoothnessConstant__()
        self.constant = self.lambda_ / self.smoothness_constant
        self.preprocess_time = time.time() - self.start_time

    def L2normProxOperator(self, x):
        """
        Compute the proximal operator for the L2 norm

        Parameters
        ----------
        x : np.ndarray of shape (num_features, 1)
            Input vector
        """

        proxed_x = np.zeros((self.num_features, 1))
        for i in range(self.num_blocks):
            block = self.blocks[i]
            norm = np.linalg.norm(x[block])

            if norm > self.constant:
                proxed_x[block] = (1 - (self.constant / norm)) * x[block]
            else:
                proxed_x[block] = np.zeros_like(x[block])

        return proxed_x

    def fista(self, max_iterations=1000, iterator=None, gamma=None):
        """
        Run the FISTA algorithm to solve the Group Lasso problem

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations to run the algorithm
        iterator : np.ndarray of shape (num_features, 1)
            Initial iterate of the optimization algorithm

        Returns
        -------
        final_iterator : np.ndarray of shape (num_features, 1)
            Final iterate of the optimization algorithm
        cache : dict
            Dictionary containing the following keys:
                - iterates : list of np.ndarray
                    List of all iterates of the optimization algorithm
                - losses : list of float
                    List of losses at each iterate of the optimization algorithm
                - time_itr : list of float
                    List of time taken at each iterate of the optimization algorithm
                - final_loss : float
                    Loss at the final iterate of the optimization algorithm
        """

        if iterator is None:
            iterator = np.random.randn(self.num_features, 1)
        if gamma is None:
            gamma = np.random.randn(self.num_features, 1)

        iterates = [iterator.copy()]
        losses = []
        time_itr = []
        theta_k = lambda k: 2 / (k + 1)

        for i in tqdm(range(max_iterations), desc="FISTA"):
            start = time.time()
            losses.append(self.__loss__(iterator))
            theta = theta_k(i + 1)

            y = iterator + theta * (gamma - iterator)
            gradient = self.__gradient__(y)
            y = y - ((1 / self.smoothness_constant) * gradient)
            updated_iterator = self.L2normProxOperator(y)

            gamma = iterator + (1 / theta) * (updated_iterator - iterator)
            iterator = updated_iterator

            iterates.append(iterator.copy())
            time_itr.append(time.time() - start)

            if np.linalg.norm(iterates[-1] - iterates[-2]) < self.tolerance:
                print("Early stopping at iteration", i)
                break

            if self.verbose:
                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {losses[-1]}")

            if self.time_crit is not None:
                if time.time() - self.start_time > self.time_crit:
                    print("Time limit reached")
                    break

        cache = {
            "iterates": iterates,
            "losses": losses,
            "time_itr": time_itr,
            "final_loss": self.__loss__(iterator),
        }
        return iterator, cache

    def __loss__(self, iterator):
        """
        Compute the Lasso objective function value at the current iterate

        Returns
        -------
        loss : float
            Lasso objective function value at the current iterate
        """

        loss = 0.5 * np.linalg.norm(self.design_matrix @ iterator - self.response) ** 2

        for block_idx in range(self.num_blocks):
            block_indices = self.blocks[block_idx]
            loss += self.lambda_ * np.linalg.norm(iterator[block_indices])

        return loss

    def __gradient__(self, iterator):
        """
        Compute the gradient of the Lasso objective function at the current iterate

        Parameters
        ----------
        iterator : np.ndarray of shape (num_features, 1)
            Current iterate of the optimization algorithm

        Returns
        -------
        gradient : np.ndarray of shape (num_features, 1)
            Gradient of the Lasso objective function at the current iterate
        """

        gradient = self.A_transpose_A @ iterator - self.A_transpose_b
        return gradient

    def __getSmoothnessConstant__(self):
        """
        Compute the smoothness constant of {1\over 2}||Ax-b||^2
        which is the largest eigenvalue of A^TA

        Returns
        -------
        smoothness_constant : float
            Smoothness constant of the objective function
        """

        return np.real(np.linalg.eigvals(self.A_transpose_A).max())
