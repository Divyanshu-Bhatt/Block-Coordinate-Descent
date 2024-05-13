import numpy as np


class ljyLDataset(object):
    """
    Simulate 15 groups independent standard Gaussian random variables. The first five groups are
    of a size 5 each, and the last 10 groups contain 500 variables each. The responses are constructed
    in a similar way as in yl1L
    """

    def __init__(self, num_observations, SNR=1.8):
        """
        Initialize the dataset

        Parameters
        ----------
        num_observations : int
            Number of observations in the dataset
        SNR : float
            Signal to Noise ratio for the gaussian noise added to the response vector
        expected_segment_density : float
            Probability of a segment being selected in the sparse linear combination
        expected_col_density : float
            Probability of a column being selected in the design matrix if it corresponds to a selected segment
        """

        self.block_indices = None
        self.design_matrix = self.__getDesignMatrix__(num_observations)
        self.x_star = self.__getRandomSparseLinearCombination__()
        self.response = self.__getResponseVector__(SNR)

    def __getDesignMatrix__(self, num_observations):
        """
        Generate the design matrix for the dataset

        Parameters
        ----------
        num_observations : int
            Number of observations in the dataset

        Returns
        -------
        design_matrix : np.ndarray of shape (num_observations, 5025)
            Design matrix for the dataset
        """

        design_matrix = []
        self.block_indices = []

        for i in range(5):
            self.block_indices.append(np.arange(i * 5, (i + 1) * 5))
        for i in range(10):
            self.block_indices.append(np.arange(25 + i * 500, 25 + (i + 1) * 500))

        # Generating the block variables from a centered multivariate Gaussian distribution
        # as described in yl1L datase
        for i in range(5):
            design_submatrix = np.random.multivariate_normal(
                mean=np.zeros(5),
                cov=np.eye(5),
                size=num_observations,
            )
            design_matrix.append(design_submatrix)

        for i in range(10):
            design_submatrix = np.random.multivariate_normal(
                mean=np.zeros(500),
                cov=np.eye(500),
                size=num_observations,
            )

            design_matrix.append(design_submatrix)

        return np.concatenate(design_matrix, axis=1)

    def __getRandomSparseLinearCombination__(
        self, expected_segment_density=0.1, expected_col_density=0.1
    ):
        """
        Generate a random sparse linear combination of segments for the dataset i.e.
        the optimum vector x* to get the response

        Parameters
        ----------
        expected_segment_density : float
            Probability of a segment being selected
        expected_col_density : float
            Probability of a column being selected in the design matrix if it corresponds to a selected segment

        Returns
        -------
        x_star : np.ndarray of shape (5025, 1)
            Optimum vector x* to get the response
        """

        # Randomly selecting segments according to the given density
        segment_indices = np.random.binomial(1, expected_segment_density, size=15)

        # Randomly selecting columns and setting the values of the optimum vector x*
        # to get a sparse linear combination of the columns of the design matrix
        x_star = np.zeros(5025)
        for i, segment in enumerate(segment_indices):
            if segment:
                block = self.block_indices[i]
                segment_cols = np.random.binomial(
                    1, expected_col_density, size=len(block)
                )
                x_star[block] = segment_cols * np.random.randn(len(block))

        return x_star.reshape(-1, 1)

    def __getResponseVector__(self, SNR):
        """
        Generate the response vector for the dataset

        Parameters
        ----------
        SNR : float
            Signal-to-noise ratio of the response vector to be generated

        Returns
        -------
        response : np.ndarray of shape (num_observations, 1)
            Response vector for the dataset
        """

        response = self.design_matrix @ self.x_star

        # Finding standard deviation of the noise to be added to the response vector
        # to get the desired signal-to-noise ratio defiend as SNR = (E[response^2] / E[noise^2])
        noise_var = (np.var(response) + np.mean(response) ** 2) / SNR
        response += np.random.normal(scale=np.sqrt(noise_var), size=response.shape)

        return response
