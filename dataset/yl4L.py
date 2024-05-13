import numpy as np
from statistics import NormalDist


class yl4LDataset(object):
    """
    110 latent variables are simulated in the same way as the third data set in [33].
    The first 50 variables contribute 3 columns each in the design matrix A with the
    i-th column among the three containing the i-th power of the variable. The next 50
    variables are encoded in a set of 3, and the final 10 variables are encoded in a set
    of 50, similar to yl1L. In addition, 4 groups of 1000 Gaussian random numbers are also
    added to A. The responses are constructed in a similar way as in yl1L.
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

        # Finding the values of the inverse cdf of the normal distribution at 0.001
        # intervals between 0 and 1 (inclusive) for quantizing the latent variables
        NORMAL_DISTRIBUTION = NormalDist(mu=0, sigma=1)

        NORMAL_INV_CDF_3 = [NORMAL_DISTRIBUTION.inv_cdf(i / 3) for i in range(1, 3)]
        NORMAL_INV_CDF_3 = [-np.inf] + NORMAL_INV_CDF_3 + [np.inf]

        NORMAL_INV_CDF_50 = [NORMAL_DISTRIBUTION.inv_cdf(i / 50) for i in range(1, 50)]
        NORMAL_INV_CDF_50 = [-np.inf] + NORMAL_INV_CDF_50 + [np.inf]

        # Generating the latent variables from a centered multivariate Gaussian distribution
        # as described in the yl1L dataset
        latent_variables = np.random.multivariate_normal(
            mean=np.zeros(110),
            cov=np.power(
                0.5, np.abs(np.subtract.outer(np.arange(110), np.arange(110)))
            ),
            size=num_observations,
        )

        # Quantising the first 50 latent variables to {0, 1, 2}
        latent_variables[:, 50:100] = (
            np.digitize(latent_variables[:, 50:100], bins=NORMAL_INV_CDF_3) - 1
        )

        # Quantising the last 10 latent variables to {0, ..., 49}
        latent_variables[:, 100:] = (
            np.digitize(latent_variables[:, 100:], bins=NORMAL_INV_CDF_50) - 1
        )

        # Getting the block indices corresponding to each latent variable
        self.block_indices = []
        for i in range(100):
            self.block_indices.append(np.arange(i * 3, (i + 1) * 3))
        for i in range(10):
            self.block_indices.append(np.arange(300 + i * 50, 300 + (i + 1) * 50))
        for i in range(4):
            self.block_indices.append(np.arange(800 + i * 1000, 800 + (i + 1) * 1000))

        # Generating the design matrix
        design_matrix = []
        for z in latent_variables[:, :50].T:
            z = z.reshape(-1, 1)
            design_submatrix = z ** np.arange(1, 4)
            design_matrix.append(design_submatrix)

        for z in latent_variables[:, 50:100].T:
            z = z.astype(int)
            design_submatrix = np.zeros((num_observations, 3))
            design_submatrix[np.arange(num_observations), z] = 1
            design_matrix.append(design_submatrix)

        for z in latent_variables[:, 100:].T:
            z = z.astype(int)
            design_submatrix = np.zeros((num_observations, 50))
            design_submatrix[np.arange(num_observations), z] = 1
            design_matrix.append(design_submatrix)

        # Adding 4 groups of 1000 Gaussian random numbers to the design matrix
        covariance_matrix = np.power(
            0.5, np.abs(np.subtract.outer(np.arange(1000), np.arange(1000)))
        )
        for i in range(4):
            design_submatrix = np.random.multivariate_normal(
                mean=np.zeros(1000),
                cov=covariance_matrix,
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
        x_star : np.ndarray of shape (4800, 1)
            Optimum vector x* to get the response
        """

        # Randomly selecting segments according to the given density
        segment_indices = np.random.binomial(1, expected_segment_density, size=114)

        # Randomly selecting columns and setting the values of the optimum vector x*
        # to get a sparse linear combination of the columns of the design matrix
        x_star = np.zeros(4800)
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
