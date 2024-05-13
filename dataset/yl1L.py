import numpy as np
from statistics import NormalDist


class yl1LDataset(object):
    """
    50 latent variables Z_1, ...,Z_50 are simulated from a centered multivariate Gaussian distribution
    with covariance between Z_i and Z_j being 0.5^(|i-j|). The first 47 latent variables are encoded in
    {0, ..., 9} according to their inverse cdf values as done in "Model selection and estimation in
    regression with grouped variables". The last three variables are encoded in {0, ..., 999}. Each
    latent variable corresponds to one segment and contributes L columns in the design matrix with each
    column j containing values of the indicator function I(Z_i = j). L is the size of the encoding set
    for Zi. The responses are a linear combination of a sparse selection of the segments plus a Gaussian
    noise.
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
        NORMAL_INV_CDF = [NORMAL_DISTRIBUTION.inv_cdf(i / 1000) for i in range(1, 1000)]
        NORMAL_INV_CDF = [-np.inf] + NORMAL_INV_CDF + [np.inf]

        # Generating the latent variables from a centered multivariate Gaussian distribution
        # as described in the before the class definition
        latent_variables = np.random.multivariate_normal(
            mean=np.zeros(50),
            cov=np.power(0.5, np.abs(np.subtract.outer(np.arange(50), np.arange(50)))),
            size=num_observations,
        )  # (num_observations, 50)

        # Quantising the first 47 latent variables to {1, ..., 10}
        latent_variables[:, :47] = np.digitize(
            latent_variables[:, :47],
            bins=([NORMAL_INV_CDF[i * 100] for i in range(0, 11)]),
        )

        # Quantising the last three latent variables to {1, ..., 1000}
        latent_variables[:, 47:] = np.digitize(
            latent_variables[:, 47:], bins=NORMAL_INV_CDF
        )

        # To make the values in {0, ..., 9} and {0, ..., 999}
        latent_variables = latent_variables - 1

        # Getting the block indices corresponding to each latent variable
        self.block_indices = []
        for i in range(47):
            self.block_indices.append(np.arange(i * 10, (i + 1) * 10))
        for i in range(3):
            self.block_indices.append(np.arange(470 + i * 1000, 470 + (i + 1) * 1000))

        # Constructing the design matrix with entry of column as I(Z_i = j) for each latent variable
        design_matrix = []
        for z in latent_variables[:, :47].T:
            z = z.astype(np.int32)
            design_submatrix = np.zeros((num_observations, 10))
            design_submatrix[np.arange(num_observations), z] = 1
            design_matrix.append(design_submatrix)

        for z in latent_variables[:, 47:].T:
            z = z.astype(int)
            design_submatrix = np.zeros((num_observations, 1000))
            design_submatrix[np.arange(num_observations), z] = 1
            design_matrix.append(design_submatrix)

        design_matrix = np.concatenate(design_matrix, axis=1)
        return design_matrix

    def __getRandomSparseLinearCombination__(
        self, expected_segment_density, expected_col_density
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
        x_star : np.ndarray of shape (3470, 1)
            Optimum vector x* to get the response
        """

        # Randomly selecting segments according to the given density
        segment_indices = np.random.binomial(1, expected_segment_density, size=50)

        # Randomly selecting columns and setting the values of the optimum vector x*
        # to get a sparse linear combination of the columns of the design matrix
        x_star = np.zeros(3470)
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
