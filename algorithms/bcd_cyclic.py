import numpy as np
import time
from tqdm import tqdm


class CyclicBlockCoordinateDescent(object):
    """
    Block Coordinate Descent algorithm for solving the Lasso problem
    in a cyclic manner, i.e. taking gradient steps in each block in a
    cyclic order.
    """

    def __init__(
        self,
        blocks,
        design_matrix,
        response,
        lambda_,
        step_size,
        tolerance=1e-10,
        verbose=False,
        time_crit=None,
    ):
        """
        Initialize the BlockCoordinateDescentRandom object

        Parameters
        ----------
        blocks : list of np.ndarray
            List of indices of the features corresponding to each block
        design_matrix : np.ndarray of shape (num_observations, num_features)
            Design matrix for the dataset
        response : np.ndarray of shape (num_observations, 1)
            Response vector for the dataset
        lambda_ : float
            Regularization parameter for Lasso
        step_size : float
            Learning rate for the gradient steps
        tolerance : float
            Tolerance for the stopping criterion of the optimization algorithm
        verbose : bool
            Verbosity flag
        time_crit : float
            Time limit for the optimization algorithm
        """

        self.start_time = time.time()
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self.design_matrix = design_matrix
        self.response = response
        self.lambda_ = lambda_
        self.step_size = step_size
        self.num_observations, self.num_features = design_matrix.shape
        self.tolerance = tolerance
        self.time_crit = time_crit
        self.verbose = verbose
        self.preprocess_time = time.time() - self.start_time

    def blockCoordinateDescentEpoch(
        self, iterator, iterates=[], losses=[], time_itr=[]
    ):
        """
        Run the Block Coordinate Descent algorithm for solving the Lasso
        problem in a cyclic manner i.e. taking gradient steps in each block
        in a cyclic order for one epoch.

        Algorithm runs in: O(num_observations * num_features * num_blocks)

        Parameters
        ----------
        iterator : np.ndarray of shape (num_features, 1)
            Current iterate of the optimization algorithm
        iterates : list of np.ndarray
            List of all iterates of the optimization algorithm
        losses : list of float
            List of losses at each iterate of the optimization algorithm
        time_itr : list of float
            List of time taken at each iterate of the optimization algorithm

        Returns
        -------
        final_iterator : np.ndarray of shape (num_features, 1)
            Optimal solution to the Lasso problem after one epoch
        iterates : list of np.ndarray
            List of all iterates of the optimization algorithm
        losses : list of float
            List of losses at each iterate of the optimization algorithm
        time_itr : list of float
            List of time taken at each iterate of the optimization algorithm
        """

        for block_idx in range(self.num_blocks):
            start = time.time()
            gradient = self.__fullGradientCalculator__(block_idx, iterator)
            iterator = self.__gradientStep__(
                block_idx, iterator, gradient, self.step_size
            )
            iterates.append(iterator.copy())
            losses.append(self.__loss__(iterator))
            time_itr.append(time.time() - start)

        return iterator, iterates, losses, time_itr

    def blockCoordinateDescentEpochFast(
        self, iterator, iterates=[], losses=[], time_itr=[]
    ):
        """
        Run the Block Coordinate Descent algorithm for solving the Lasso
        problem in a cyclic manner i.e. taking gradient steps in each block
        in a cyclic order and calculating the current block's gradient,
        using the previous block's gradient for one epoch.

        Algorithm runs in: O(num_observations * num_features + (num_features * block_size) * (num_blocks - 1))

        Parameters
        ----------
        iterator : np.ndarray of shape (num_features, 1)
            Current iterate of the optimization algorithm
        iterates : list of np.ndarray
            List of all iterates of the optimization algorithm
        losses : list of float
            List of losses at each iterate of the optimization algorithm
        time_itr : list of float
            List of time taken at each iterate of the optimization algorithm

        Returns
        -------
        final_iterator : np.ndarray of shape (num_features, 1)
            Optimal solution to the Lasso problem after one epoch
        iterates : list of np.ndarray
            List of all iterates of the optimization algorithm
        losses : list of float
            List of losses at each iterate of the optimization algorithm
        time_itr : list of float
            List of time taken at each iterate of the optimization algorithm
        """

        start = time.time()
        # Calculating the summation term for the given iterator
        # S^(k,0) = A @ x^(k,0) - b
        summation_term = (
            self.design_matrix @ iterator - self.response
        )  # O (num_observations * num_features)

        # Calculating the gradient with respect to the zeroth block of features
        # \nabla_0 f(x^(k,0)) = (A^T)_0 @ S^(k,0) + lambda * x_0/|x_0|
        gradient = self.design_matrix.T[self.blocks[0]] @ summation_term
        gradient += self.lambda_ * (
            iterator[self.blocks[0]] / (np.linalg.norm(iterator[self.blocks[0]]) + 1e-8)
        )  # O (num_observations * block_size)

        # Taking a gradient step in the zeroth block of features
        # x^(k,1)_0 = x^(k,0)_0 - \alpha * \nabla_0 f(x^(k,0))
        iterator[self.blocks[0]] -= self.step_size * gradient  # O (block_size)
        iterates.append(iterator.copy())
        losses.append(self.__loss__(iterator))
        time_itr.append(time.time() - start)

        for block_idx in range(1, self.num_blocks):
            start = time.time()

            # Updating the summation term for the next block of features
            # S^(k+1,p) = S^(k,p-1) - \alpha * A[:, p] @ \nabla_{p-1} f(x^(k,p-1))
            summation_term = (
                summation_term
                - self.step_size
                * self.design_matrix[:, self.blocks[block_idx - 1]]
                @ gradient
            )  # O (num_observations * block_size)

            # Calculating the gradient with respect to the p-th block of features
            # \nabla_p f(x^(k,p)) = (A^T)_p @ S^(k,p) + lambda * x_p/|x_p|
            gradient = self.design_matrix.T[self.blocks[block_idx]] @ summation_term
            gradient += self.lambda_ * (
                iterator[self.blocks[block_idx]]
                / (np.linalg.norm(iterator[self.blocks[block_idx]]) + 1e-8)
            )  # O (num_observations * block_size)

            # Taking a gradient step in the p-th block of features
            # x^(k,p+1)_p = x^(k,p)_p - \alpha * \nabla_p f(x^(k,p))
            iterator[self.blocks[block_idx]] -= (
                self.step_size * gradient
            )  # O (block_size)

            iterates.append(iterator.copy())
            losses.append(self.__loss__(iterator))
            time_itr.append(time.time() - start)

        # x^(k+1,0) = x^(k,J) where J is the number of blocks
        return iterator, iterates, losses, time_itr

    def blockCoordinateDescent(self, max_epochs=1000, iterator=None, algo_flag=True):
        """
        Run the Block Coordinate Descent algorithm for solving the Lasso
        problem in a cyclic manner i.e. taking gradient steps in each block
        in a cyclic order.

        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs to run the optimization algorithm
            if the stopping criterion is not met
        iterator : np.ndarray of shape (num_features, 1)
            Initial iterate of the optimization algorithm
        algo_flag : bool
            Flag to whether run the fast version of the algorithm or not

        Returns
        -------
        final_iterator : np.ndarray of shape (num_features, 1)
            Optimal solution to the Lasso problem
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
        iterates = [iterator.copy()]
        time_itr = []
        losses = []

        if algo_flag:
            descent_epoch_algo = self.blockCoordinateDescentEpochFast
        else:
            descent_epoch_algo = self.blockCoordinateDescentEpoch

        for i in tqdm(range(max_epochs), desc="Block Coordinate Descent"):
            iterator, iterates, losses, time_itr = descent_epoch_algo(
                iterator, iterates, losses, time_itr
            )

            if self.verbose:
                if i % 5 == 0:
                    print(
                        f"Epoch {i}| Loss: {losses[-1]}| Stopping Criteria: {np.linalg.norm(iterates[-1] - iterates[-2])}"
                    )

            if np.linalg.norm(iterates[-1] - iterates[-2]) < self.tolerance:
                print("Early stopping at iteration", i)
                break

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

    def __fullGradientCalculator__(self, block_idx, iterator):
        """
        Compute the gradient of the Lasso objective function with respect to
        the block of features indexed by block_idx at the current iterate

        Parameters
        ----------
        block_idx : int
            Index of the block of features
        iterator : np.ndarray of shape (num_features, 1)
            Current iterate of the optimization algorithm

        Returns
        -------
        gradient : np.ndarray of shape (block_size, 1)
            Gradient of the Lasso objective function with respect to the block
            of features indexed by block_idx at the current iterate
        """

        block_indices = self.blocks[block_idx]

        # Gradient with respect to the block of features indexed by block_idx
        # at current iterate for first term in the gradient formula
        summation_term = (
            self.design_matrix @ iterator - self.response
        )  # O(num_observations * num_features)
        gradient = (
            self.design_matrix[:, block_indices].T @ summation_term
        )  # O(num_observations * block_size)

        # Gradient with respect to the block of features indexed by block_idx
        # at current iterate for second term in the gradient formula
        gradient = gradient + self.lambda_ * (
            iterator[block_indices] / (np.linalg.norm(iterator[block_indices]) + 1e-8)
        )  # O(block_size)

        return gradient

    def __gradientStep__(self, block_idx, iterator, gradient, learning_rate):
        """
        Take a gradient step in the block of features indexed by block_idx
        in the direction of the gradient

        Parameters
        ----------
        block_idx : int
            Index of the block of features
        iterator : np.ndarray of shape (num_features, 1)
            Current iterate of the optimization algorithm
        gradient : np.ndarray of shape (block_size, 1)
            Gradient of the Lasso objective function with respect to the block
            of features indexed by block_idx at the current iterate
        learning_rate : float
            Learning rate for the gradient step

        Returns
        -------
        updated_iterator : np.ndarray of shape (num_features, 1)
            Updated iterate after taking a gradient step in the block of features
            indexed by block_idx
        """

        block_indices = self.blocks[block_idx]
        updated_iterator = iterator.copy()
        updated_iterator[block_indices] = (
            iterator[block_indices] - learning_rate * gradient
        )

        return updated_iterator
