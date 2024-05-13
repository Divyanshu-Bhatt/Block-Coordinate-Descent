import numpy as np
from tqdm import tqdm
import time


class BlockCoordinateDescentGroupLasso(object):
    """
    Block Coordinate Descent algorithm for solving the Group Lasso problem
    According to the paper "Efficient Block-coordinate Descent Algorithms for the Group Lasso"
    by Zhiwei (Tony) Qin, Katya Scheinberg, and Donald Goldfarb, Algorithm 2.1
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
        self.num_observations, self.num_features = design_matrix.shape
        self.tolerance = tolerance
        self.verbose = verbose
        self.time_crit = time_crit

        self.blocks_complement = self.__getBlockIndicesComplement__()
        self.Mj_matrices = self.__getMjMatrices__()
        self.eigendecomposition = self.__blockwiseEigenDecomposition__()

        self.preprocess_time = time.time() - self.start_time

    def blockCoordinateDescentEpoch(
        self, iterator, iterates=[], losses=[], time_itr=[]
    ):
        """
        Run the Block Coordinate Descent algorithm for solving the Group Lasso
        problem according to Algorithm 2.2 in the paper by Qin et al. for one epoch

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
            Optimal solution to the Group Lasso problem after one epoch
        iterates : list of np.ndarray
            List of all iterates of the optimization algorithm
        losses : list of float
            List of losses at each iterate of the optimization algorithm
        time_itr : list of float
            List of time taken at each iterate of the optimization algorithm
        """

        for block_idx in range(self.num_blocks):
            start = time.time()
            block_indices = self.blocks[block_idx]
            remaining_indices = self.blocks_complement[block_idx]

            dummy = (
                self.design_matrix[:, remaining_indices] @ iterator[remaining_indices]
            )
            d = dummy - self.response
            p = (self.design_matrix[:, block_indices]).T @ d

            if np.linalg.norm(p) <= self.lambda_:
                iterator[block_indices] = 0
            else:
                pj = (self.design_matrix[:, block_indices]).T @ dummy

                delta = self.__getPhiRootsNewtonRapson__(block_idx, pj)
                y_j = self.__getyj__(block_idx, delta, pj)
                iterator[block_indices] = delta * y_j

            iterates.append(iterator.copy())
            losses.append(self.__loss__(iterator))
            time_itr.append(time.time() - start)
        return iterator, iterates, losses, time_itr

    def blockCoordinateDescent(self, max_epochs=1000, iterator=None):
        """
        Run the Block Coordinate Descent algorithm for solving the Group Lasso
        problem according to Algorithm 2.1 in the paper by Qin et al.

        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs for the optimization algorithm
        iterator : np.ndarray of shape (num_features, 1)
            Initial iterate of the optimization algorithm

        Returns
        -------
        final_iterator : np.ndarray of shape (num_features, 1)
            Optimal solution to the Group Lasso problem
        iterates : list of np.ndarray
            List of iterates at each step of the optimization algorithm
        losses : list of float
            List of loss values at each step of the optimization algorithm
        """

        if iterator is None:
            iterator = np.random.randn(self.num_features, 1)
        iterates = [iterator.copy()]
        losses = []
        time_itr = []

        for i in tqdm(range(max_epochs), desc="Block Coordinate Descent"):
            iterator, iterates, losses, time_itr = self.blockCoordinateDescentEpoch(
                iterator, iterates, losses, time_itr
            )

            if self.verbose:
                if i % 1 == 0:
                    print(f"Epoch {i}| Loss: {losses[-1]}")

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

    def __getPhiRootsNewtonRapson__(self, block_idx, pj):
        """
        Calculates the roots of the function \phi(\delta) as defined in the equation
        (8) in the paper by Qin et al. using Newton-Rapson method

        Parameters
        ----------
        block_idx : int
            Index of the block for which the roots are to be found
        pj : np.ndarray of shape (block_size, 1)
            pj vector for the block calculated in the blockCoordinateDescent function

        Returns
        -------
        delta : float
            Optimal value of \delta for the block
        """

        delta = np.random.randn()

        for _ in range(100000):
            phi, gradient_phi = self.__phiCalculator__(block_idx, delta, pj)
            delta -= phi / (gradient_phi + 1e-8)

            # Stopping criterion
            if abs(phi) < 1e-10:
                break

        if abs(phi) > 1e-10:
            print("Newton-Rapson did not converge")

        return delta

    def __phiCalculator__(self, block_idx, delta, pj):
        """
        Calculates the value of the function \phi(\delta) as defined in the equation
        (8) in the paper by Qin et al.

        Parameters
        ----------
        block_idx : int
            Index of the block for which the value of phi is to be calculated
        delta : float
            Value of the delta for which the value of phi is to be calculated
        pj : np.ndarray of shape (block_size, 1)
            pj vector for the block calculated in the blockCoordinateDescent function

        Returns
        -------
        phi : float
            Value of \phi(\delta) for the given block and delta
        gradient_phi : float
            Value of the gradient of \phi(\delta) for the given block and delta
        """

        y_j_norm_squared, gradient_y_j_norm_squared = self.__yjNormSquaredCalculator__(
            block_idx, delta, pj
        )

        phi = 1 - (1 / (np.sqrt(y_j_norm_squared) + 1e-8))
        gradient_phi = 0.5 * ((y_j_norm_squared) ** (-1.5)) * gradient_y_j_norm_squared

        return phi, gradient_phi

    def __yjNormSquaredCalculator__(self, block_idx, delta, pj):
        """
        Calculating the value of norm(y_j)^2 for the given block and delta as defined
        in the equation (7) in the paper by Qin et al.

        Parameters
        ----------
        block_idx : int
            Index of the block for which the value of y_j^2 is to be calculated
        delta : float
            Value of the delta for which the value of y_j^2 is to be calculated
        pj : np.ndarray of shape (block_size, 1)
            p vector for the block calculated in the blockCoordinateDescent function

        Returns
        -------
        y_j_norm_squared : float
            Value of y_j^2 for the given block and delta
        gradient_y_j_norm_squared : float
            Value of the gradient of y_j^2 for the given block and delta
        """

        eigenvalues_block, eigenvectors_block = self.eigendecomposition[block_idx]

        numerator = (eigenvectors_block.T @ pj) ** 2
        denominator = eigenvalues_block * delta + self.lambda_

        y_j_norm_squared = np.sum(numerator / (((denominator) ** 2) + 1e-10))
        gradient_y_j_norm_squared = -2 * np.sum(
            (numerator * eigenvalues_block) / (((denominator) ** 3) + 1e-10)
        )

        return y_j_norm_squared, gradient_y_j_norm_squared

    def __getyj__(self, block_idx, delta, pj):
        """
        Calculate the value of y_j for the given block and delta as defined in the
        equation (6) in the paper by Qin et al.

        Parameters
        ----------
        block_idx : int
            Index of the block for which the value of y_j is to be calculated
        delta : float
            Value of the delta for which the value of y_j is to be calculated
        pj : np.ndarray of shape (block_size, 1)
            pj vector for the block calculated in the blockCoordinateDescent function

        Returns
        -------
        y_j : np.ndarray of shape (block_size, 1)
            Value of y_j for the given block and delta
        """

        Mj_matrix = self.Mj_matrices[block_idx]
        y_j = (delta * Mj_matrix) + (self.lambda_ * np.eye(Mj_matrix.shape[0]))
        y_j = np.linalg.inv(y_j) @ pj
        y_j = y_j * (-1)

        return y_j

    def __getMjMatrices__(self):
        """
        Calculate the matrices M_j = A_j^TA_j for each block j

        Returns
        -------
        Mj_matrices : list of np.ndarray
            List of matrices M_j = A_j^TA_j for each block j
        """

        Mj_matrices = []
        for block in self.blocks:
            design_matrix_block = self.design_matrix[:, block]
            Mj_matrices.append(design_matrix_block.T @ design_matrix_block)

        return Mj_matrices

    def __blockwiseEigenDecomposition__(self):
        """
        Calculate the eigenvalues and eigenvectors for each block M_j = A_j^TA_j

        Returns
        -------
        eigendecomposition : list of tuple of np.ndarray
            List of eigenvalues and eigenvectors for each block M_j = A_j^TA_j
        """

        eigendecomposition = []
        for block_idx in range(self.num_blocks):
            eigenvalues, eigenvectors = np.linalg.eig(self.Mj_matrices[block_idx])
            eigenvalues = eigenvalues.reshape(-1, 1)
            eigenvectors = eigenvectors.astype(np.float64)

            eigendecomposition.append((eigenvalues, eigenvectors))

        return eigendecomposition

    def __getBlockIndicesComplement__(self):
        """
        Get the complement of the block indices i.e. the indices of the features not
        in the block

        Returns
        -------
        remaining_block_indices : list of np.ndarray
            List of indices of the features not in the block
        """

        remaining_block_indices = []
        for _, block in enumerate(self.blocks):
            remaining_indices = np.setdiff1d(np.arange(self.num_features), block)
            remaining_block_indices.append(remaining_indices)

        return remaining_block_indices
