import numpy as np


class mgb2LDataset(object):
    """
    5001 variables are simulated as in yl1L without categorization. They are then divided
    into six groups, with first containing one variable and the rest containing 1000 each.
    The responses are constructed in a similar way as in yl1L. We collect 2000 observations.
    """

    def __init__(
        self,
        num_observations,
        SNR=1.8,
        expected_segment_density=0.1,
        expected_col_density=0.1,
    ):
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
        self.x_star = self.__getRandomSparseLinearCombination__(
            expected_segment_density, expected_col_density
        )
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
        design_matrix : np.ndarray of shape (num_observations, 3470)
            Design matrix for the dataset
        """

        # Generating the latent variables from a centered multivariate Gaussian distribution
        # as described in the yl1L dataset
        design_matrix = np.random.multivariate_normal(
            mean=np.zeros(5001),
            cov=np.power(
                0.5, np.abs(np.subtract.outer(np.arange(5001), np.arange(5001)))
            ),
            size=num_observations,
        )

        self.block_indices = []
        self.block_indices.append(np.arange(1))
        for i in range(5):
            self.block_indices.append(np.arange(i * 1000 + 1, (i + 1) * 1000 + 1))

        return design_matrix

    def __getRandomSparseLinearCombination__(
        self, expected_segment_density=0.1, expected_col_density=0.1
    ):
        """
        Generate a random sparse linear combination of the variables

        Parameters
        ----------
        expected_segment_density : float
            Probability of a segment being selected
        expected_col_density : float
            Probability of a column being selected in the design matrix if it corresponds to a selected segment

        Returns
        -------
        x_star : np.ndarray of shape (5001, 1)
            Sparse linear combination of the variables
        """

        # Randomly selecting segments according to the given density
        segment_indices = np.random.binomial(1, expected_segment_density, size=6)

        # Randomly selecting non-zero coefficients for the selected segments
        x_star = np.zeros(5001)
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
