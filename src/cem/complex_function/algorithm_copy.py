from turtle import update
import numpy as np


import torch
import torch.nn as nn
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors
import sys


from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.pyplot import figure


class CovMatrix:
    def __init__(self, **kwargs):
        """centroid: torch.Tensor = kwargs['centroid']
        sigma = kwargs['sigma']
        diag_covMatrix= kwargs.get('diag_covMatrix', False)  #default value of diag_covMatrix = False
        noise_multiplier = kwargs['noise_multiplier']

        """
        centroid: torch.Tensor = kwargs['centroid']
        sigma = kwargs['sigma']
        # default value of diag_covMatrix = False
        diag_covMatrix = kwargs.get('diag_covMatrix', False)
        noise_multiplier = kwargs['noise_multiplier']

        policy_dim = centroid.size()[0]
        self.policy_dim = policy_dim
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) *
                              torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier
        self.diag_covMatrix = diag_covMatrix

    def get_cov(self):
        return torch.clone(self.cov)

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):
        if self.diag_covMatrix:
            # Only use the diagonal of the covariance matrix when the matrix of covariance has not inverse
            policy_dim = centroid.size()[0]
            param_noise = torch.randn(pop_size, policy_dim)
            weights = centroid + param_noise * \
                torch.sqrt(torch.diagonal(self.cov))
        else:
            # The params of policies at iteration t+1 are drawn according to a multivariate
            # Gaussian whose center is centroid and whose shape is defined by cov
            dist = torch.distributions.MultivariateNormal(
                centroid, covariance_matrix=self.cov
            )
            weights = [dist.sample() for _ in range(pop_size)]
        return weights

    def update_covariance(self, elite_weights) -> None:

        self.cov = torch.cov(elite_weights.T) + self.noise

    def update_covariance_inverse(self, elite_weights) -> None:
        def inv_diagonal(mat: torch.Tensor) -> torch.Tensor:
            res = torch.zeros(self.policy_dim, self.policy_dim)
            for i in range(self.policy_dim):
                if self.cov[i, i] == 0:
                    raise Exception(
                        "Tried to invert 0 in the diagonal of the cov matrix")
                res[i][i] = 1/self.cov[i][i]
            return res

        cov = torch.cov(elite_weights.T) + self.noise
        if self.diag_covMatrix:
            self.cov = inv_diagonal(torch.diag(torch.diag(cov))) + self.noise
        else:
            u = torch.linalg.cholesky(cov)
            self.cov = torch.cholesky_inverse(u) + self.noise

    def update_covariance_inverse_resize(self, elite_weights) -> None:
        cov = torch.cov(elite_weights.T)
        u = torch.linalg.cholesky(cov)
        cov_i = torch.cholesky_inverse(u)

        eig_cov_max = torch.linalg.eigh(cov)[0][-1]
        eig_cov_i_max = torch.linalg.eigh(cov_i)[0][-1]
        self.cov = (cov_i*eig_cov_max/eig_cov_i_max) + self.noise

    def update_cov_circle(self, elite_weights):
        cov = torch.cov(elite_weights.T)
        u = torch.linalg.cholesky(cov)
        cov_i = torch.cholesky_inverse(u)
        eig_cov_max = torch.linalg.eigh(cov)[0][-1]
        eig_cov_i_max = torch.linalg.eigh(cov_i)[0][-1]
        facteur = eig_cov_max/eig_cov_i_max
        self.cov = torch.diag(torch.ones(self.policy_dim)
                              * facteur) + self.noise


def plot_confidence_ellipse(cov, ax, centroid, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    cov : array_like, shape (n, )
                                                                                                                                    Input matrix.

    ax : matplotlib.axes.Axes
                                                                                                                                    The axes object to draw the ellipse into.

    centroid
                                                                                                                                    The weights

    n_std : float
                                                                                                                                    The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Compute the stdandard deviation of x from
    # the square root of the variance and multiply
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # compute the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    ax.scatter([centroid[0]], [centroid[1]], c='b')

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(centroid[0], centroid[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def test_a_converge(elite_scores, seuil_convergence, delta_convergence):
    """test si l'algo a converge en regardant le plus grand score des elites et en le comparant a la valeur seuil de convergence

    retourne Vrai ou faux"""
    max_score = np.max(elite_scores)
    if seuil_convergence is None or delta_convergence is None:
        return False
    return abs(seuil_convergence-max_score) < delta_convergence


def CEM_GEN(score_function, initial_matrices:  list[CovMatrix], ls_update_covariance_functions:  list, **kwargs):
    """Generic Version of the CEM. Will execute the CEM with every initial matrice and their corresponding covariance_functions, and merge the obtained weights to only select the best individuals.


    Parameters
    ----
    score_function:
                    The function that associate a score to every weight

    initial_matrices  : list[CovMatrix]
                    The list of the initial matrices

    ls_update_covariance_functions :  list[functions]
                    List of the corresponding update functions that will update the covariance matrices, once the list of new weights are passed as argument. 

    kwargs (REQUIRED):
                    Additional keyword arguments:
                                    seed  
                                    centroid 
                                    max_epochs ( numbder of generations)
                                    pop_size 
                                    elites_nb 
                                    seuil_convergence (can be None)
                                    delta_convergence (can be None)


    Returns
    -------

    (all_weights, all_elites, all_centroids, all_covs, all_percentages) : 

                    respectively:

                    - the np arrays of all weights generated by each matrices : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]

                                    >>>	all_weights  = [ [[weights_generation_1 for mat1] ,[weights_generation_2 for mat1] , ....  ],
                                                                             [ [weights_generation_1 for mat2] ,[weights_generation_2 for mat2] , ....  ],
                                                                            [ [weights_generation_1 for mat3] ,[weights_generation_2 for mat3] , ....  ] ]

                    - same thing for the elites, centroid, covs

                    - all percentages  represents the proportion of elite individuals generated by each matrix at each generation : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]
                                                    all_percentages  = 	[ 	[percentages_mat1_gen1 , percentages_mat1_gen2 ,percentages_mat1_gen3 ,],

                                                                                                                                                    [percentages_mat2_gen1, percentages_mat2_gen2 ,percentages_mat2_gen2 ,],
                                                                                                                                                                    ....
                                                                                                                                                    [percentages_mat3_gen_1 , percentages_mat3_gen_max_epoch ,percentages_mat3_gen_max_epoch] 
                                                                                                                                                    ]"""

    seed: int = kwargs['seed']
    centroid = kwargs['centroid']
    max_epochs: int = kwargs['max_epochs']
    pop_size: int = kwargs['pop_size']
    elites_nb: int = kwargs['elites_nb']

    seuil_convergence: int = kwargs['seuil_convergence']
    delta_convergence: int = kwargs['delta_convergence']
    flag_a_converge = False
    convergence_generation = None

    all_centroids = [centroid.detach().cpu().numpy()]

    torch.manual_seed(seed)
    centroid = centroid.clone().detach()

    N = len(initial_matrices)
    size = pop_size//N
    ls_matrix = initial_matrices

    # se deduit du haut
    all_covs = [[mat.get_cov().detach().cpu().numpy()] for mat in ls_matrix]
    all_percentages = [[] for _ in ls_matrix]
    all_weights = [[] for _ in ls_matrix]
    all_elites = []
    all_elite_scores = []

    for epoch in range(max_epochs):
        print(f"Gen {epoch}/{max_epochs}")
        #print(f'Simulating {epoch}/{max_epochs} \n', end='\r')
        if flag_a_converge:
            all_centroids.append(all_centroids[-1])
            for i in range(N):
                all_covs[i].append(all_covs[i][-1])
                all_percentages[i].append(all_percentages[i][-1])
                all_weights[i].append(all_weights[i][-1])
            all_elites.append(all_elites[-1])
            all_elite_scores.append(all_elite_scores[-1])
            continue

        # The params of policies at iteration t+1 are drawn according to a multivariate
        # Gaussian whose center is centroid and whose shaoe is defined by cov

        # for matrix in liste_matrxic, generate weights
        # generated (weights,label) for each matrix, with label = index of matrix
        weights_and_labels = []
        for i in range(N):
            mat = ls_matrix[i]
            weights_mat = mat.generate_weights(centroid, size)
            all_weights[i].append([t.detach().cpu().numpy()
                                  for t in weights_mat])
            weights_mat = zip(
                weights_mat, [i for _ in range(len(weights_mat))])
            weights_and_labels += weights_mat

        for mat in ls_matrix:
            mat.update_noise()

        scores = []
        labels = []
        # generate pop_size generations
        for w, label in weights_and_labels:
            scores.append(score_function(w))
            labels.append(label)

        # Keep only best individuals to compute the new centroid
        elites_idxs = np.argsort(scores)[:elites_nb]
        elite_scores = np.array(scores)[elites_idxs]
        elites_weights = [weights_and_labels[k][0] for k in elites_idxs]
        elites_labels = [weights_and_labels[k][1] for k in elites_idxs]

        for i in range(N):
            proportion = elites_labels.count(i) / elites_nb
            all_percentages[i].append(proportion)

        all_elites.append([t.detach().cpu().numpy() for t in elites_weights])
        all_elite_scores.append(elite_scores)

        elites_weights = torch.cat(
            [torch.tensor(w).unsqueeze(0) for w in elites_weights], dim=0
        )
        centroid = elites_weights.mean(0)

        for f in ls_update_covariance_functions:
            f(elites_weights)

        all_centroids.append(centroid.detach().cpu().numpy())

        for i in range(N):
            mat = ls_matrix[i]
            all_covs[i].append(mat.get_cov().detach().cpu().numpy())

        flag_a_converge = test_a_converge(
            elite_scores, seuil_convergence, delta_convergence)
        convergence_generation = epoch if flag_a_converge else None

   
    all_weights = np.array(all_weights)
    all_elites = np.array(all_elites)
    all_elite_scores = np.array(all_elite_scores)
    all_centroids = np.array(all_centroids)
    all_covs = np.array(all_covs)
    all_percentages = np.array(all_percentages)

    output = {}
    output.update(all_weights=all_weights, all_elites=all_elites, all_elite_scores=all_elite_scores, all_centroids=all_centroids,
                  all_covs=all_covs, all_percentages=all_percentages, convergence_generation=convergence_generation)

    return output


def CEM(score_function, **kwargs):
    """Generic Version of the CEM. Will execute the CEM with every initial matrice and their corresponding covariance_functions, and merge the obtained weights to only select the best individuals.


    Parameters
    ----
    score_function:
                    The function that associate a score to every weight

    initial_matrices  : list[CovMatrix]
                    The list of the initial matrices

    ls_update_covariance_functions :  list[functions]
                    List of the corresponding update functions that will update the covariance matrices, once the list of new weights are passed as argument. 

    kwargs (REQUIRED):
                    Additional keyword arguments:
                                    seed  
                                    centroid 
                                    max_epochs ( numbder of generations)
                                    pop_size 
                                    elites_nb 

    Returns
    -------

    (all_weights, all_elites, all_centroids, all_covs, all_percentages) : 

                    respectively:

                    - the np arrays of all weights generated by each matrices : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]

                                    >>>	all_weights  = [ [[weights_generation_1 for mat1] ,[weights_generation_2 for mat1] , ....  ],
                                                                             [ [weights_generation_1 for mat2] ,[weights_generation_2 for mat2] , ....  ],
                                                                            [ [weights_generation_1 for mat3] ,[weights_generation_2 for mat3] , ....  ] ]

                    - same thing for the elites, centroid, covs

                    - all percentages  represents the proportion of elite individuals generated by each matrix at each generation : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]
                                                    all_percentages  = 	[ 	[percentages_mat1_gen1 , percentages_mat1_gen2 ,percentages_mat1_gen3 ,],

                                                                                                                                                    [percentages_mat2_gen1, percentages_mat2_gen2 ,percentages_mat2_gen2 ,],
                                                                                                                                                                    ....
                                                                                                                                                    [percentages_mat3_gen_1 , percentages_mat3_gen_max_epoch ,percentages_mat3_gen_max_epoch] 
                                                                                                                                                    ]"""
    matrix = CovMatrix(**kwargs)
    return {**CEM_GEN(score_function, [matrix], [matrix.update_covariance], **kwargs), 'list_matrix_names': ['CEM']}


def CEMi(score_function, **kwargs):
    """Generic Version of the CEM. Will execute the CEM with every initial matrice and their corresponding covariance_functions, and merge the obtained weights to only select the best individuals.


    Parameters
    ----
    score_function:
                    The function that associate a score to every weight

    initial_matrices  : list[CovMatrix]
                    The list of the initial matrices

    ls_update_covariance_functions :  list[functions]
                    List of the corresponding update functions that will update the covariance matrices, once the list of new weights are passed as argument. 

    kwargs (REQUIRED):
                    Additional keyword arguments:
                                    seed  
                                    centroid 
                                    max_epochs ( numbder of generations)
                                    pop_size 
                                    elites_nb 

    Returns
    -------

    (all_weights, all_elites, all_centroids, all_covs, all_percentages) : 

                    respectively:

                    - the np arrays of all weights generated by each matrices : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]

                                    >>>	all_weights  = [ [[weights_generation_1 for mat1] ,[weights_generation_2 for mat1] , ....  ],
                                                                             [ [weights_generation_1 for mat2] ,[weights_generation_2 for mat2] , ....  ],
                                                                            [ [weights_generation_1 for mat3] ,[weights_generation_2 for mat3] , ....  ] ]

                    - same thing for the elites, centroid, covs

                    - all percentages  represents the proportion of elite individuals generated by each matrix at each generation : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]
                                                    all_percentages  = 	[ 	[percentages_mat1_gen1 , percentages_mat1_gen2 ,percentages_mat1_gen3 ,],

                                                                                                                                                    [percentages_mat2_gen1, percentages_mat2_gen2 ,percentages_mat2_gen2 ,],
                                                                                                                                                                    ....
                                                                                                                                                    [percentages_mat3_gen_1 , percentages_mat3_gen_max_epoch ,percentages_mat3_gen_max_epoch] 
                                                                                                                                                    ]"""

    matrix = CovMatrix(**kwargs)
    return {**CEM_GEN(score_function, [matrix], [matrix.update_covariance_inverse], **kwargs), 'list_matrix_names': ['CEMi']}


def CEMir(score_function, **kwargs):
    """CEMir
    Parameters
    ----
    score_function:
                    The function that associate a score to every weight

    initial_matrices  : list[CovMatrix]
                    The list of the initial matrices

    ls_update_covariance_functions :  list[functions]
                    List of the corresponding update functions that will update the covariance matrices, once the list of new weights are passed as argument. 

    kwargs (REQUIRED):
                    Additional keyword arguments:
                                    seed  
                                    centroid 
                                    max_epochs ( numbder of generations)
                                    pop_size 
                                    elites_nb 

    Returns
    -------

    (all_weights, all_elites, all_centroids, all_covs, all_percentages,['CEMir']) : 

                    respectively:

                    - the np arrays of all weights generated by each matrices : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]

                                    >>>	all_weights  = [ [[weights_generation_1 for mat1] ,[weights_generation_2 for mat1] , ....  ],
                                                                             [ [weights_generation_1 for mat2] ,[weights_generation_2 for mat2] , ....  ],
                                                                            [ [weights_generation_1 for mat3] ,[weights_generation_2 for mat3] , ....  ] ]

                    - same thing for the elites, centroid, covs

                    - all percentages  represents the proportion of elite individuals generated by each matrix at each generation : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]
                                                    all_percentages  = 	[ 	[percentages_mat1_gen1 , percentages_mat1_gen2 ,percentages_mat1_gen3 ,],

                                                                                                                                                    [percentages_mat2_gen1, percentages_mat2_gen2 ,percentages_mat2_gen2 ,],
                                                                                                                                                                    ....
                                                                                                                                                    [percentages_mat3_gen_1 , percentages_mat3_gen_max_epoch ,percentages_mat3_gen_max_epoch] 
                                                                                                                                                    ]"""

    matrix = CovMatrix(**kwargs)
    return {**CEM_GEN(score_function, [matrix], [matrix.update_covariance_inverse_resize], **kwargs), 'list_matrix_names': ['CEMir']}


def CEM_circle(score_function, **kwargs):
    matrix = CovMatrix(**kwargs)
    return {**CEM_GEN(score_function, [matrix], [matrix.update_cov_circle], **kwargs), 'list_matrix_names': ['CEM_circle']}


def CEM_plus_CEMi(score_function, **kwargs):
    """CEM+CEMi

    Parameters
    ----
    score_function:
                    The function that associate a score to every weight

    initial_matrices  : list[CovMatrix]
                    The list of the initial matrices

    ls_update_covariance_functions :  list[functions]
                    List of the corresponding update functions that will update the covariance matrices, once the list of new weights are passed as argument. 

    kwargs (REQUIRED):
                    Additional keyword arguments:
                                    seed  
                                    centroid 
                                    max_epochs ( numbder of generations)
                                    pop_size 
                                    elites_nb 

    Returns
    -------

    (all_weights, all_elites, all_centroids, all_covs, all_percentages,['CEM','CEMi']) : 

                    respectively:

                    - the np arrays of all weights generated by each matrices : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]

                                    >>>	all_weights  = [ [[weights_generation_1 for mat1] ,[weights_generation_2 for mat1] , ....  ],
                                                                             [ [weights_generation_1 for mat2] ,[weights_generation_2 for mat2] , ....  ],
                                                                            [ [weights_generation_1 for mat3] ,[weights_generation_2 for mat3] , ....  ] ]

                    - same thing for the elites, centroid, covs

                    - all percentages  represents the proportion of elite individuals generated by each matrix at each generation : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]
                                                    all_percentages  = 	[ 	[percentages_mat1_gen1 , percentages_mat1_gen2 ,percentages_mat1_gen3 ,],

                                                                                                                                                    [percentages_mat2_gen1, percentages_mat2_gen2 ,percentages_mat2_gen2 ,],
                                                                                                                                                                    ....
                                                                                                                                                    [percentages_mat3_gen_1 , percentages_mat3_gen_max_epoch ,percentages_mat3_gen_max_epoch] 
                                                                                                                                                    ]"""
    matrix_CEM = CovMatrix(**kwargs)
    matrix_CEMi = CovMatrix(**kwargs)
    ls_matrix = [matrix_CEM, matrix_CEMi]
    ls_update_functions = [matrix_CEM.update_covariance,
                           matrix_CEMi.update_covariance_inverse]

    return {**CEM_GEN(score_function, ls_matrix, ls_update_functions, **kwargs), 'list_matrix_names': ['CEM', 'CEMi']}


def CEM_plus_CEMir(score_function, **kwargs):
    """Generic Version of the CEM. Will execute the CEM with every initial matrice and their corresponding covariance_functions, and merge the obtained weights to only select the best individuals.


    Parameters
    ----
    score_function:
                    The function that associate a score to every weight

    initial_matrices  : list[CovMatrix]
                    The list of the initial matrices

    ls_update_covariance_functions :  list[functions]
                    List of the corresponding update functions that will update the covariance matrices, once the list of new weights are passed as argument. 

    kwargs (REQUIRED):
                    Additional keyword arguments:
                                    seed  
                                    centroid 
                                    max_epochs ( numbder of generations)
                                    pop_size 
                                    elites_nb 

    Returns
    -------

    (all_weights, all_elites, all_centroids, all_covs, all_percentages) : 

                    respectively:

                    - the np arrays of all weights generated by each matrices : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]

                                    >>>	all_weights  = [ [[weights_generation_1 for mat1] ,[weights_generation_2 for mat1] , ....  ],
                                                                             [ [weights_generation_1 for mat2] ,[weights_generation_2 for mat2] , ....  ],
                                                                            [ [weights_generation_1 for mat3] ,[weights_generation_2 for mat3] , ....  ] ]

                    - same thing for the elites, centroid, covs

                    - all percentages  represents the proportion of elite individuals generated by each matrix at each generation : 

                                    >>> for initial_matrices  =  [mat1, mat2, mat3]
                                                    all_percentages  = 	[ 	[percentages_mat1_gen1 , percentages_mat1_gen2 ,percentages_mat1_gen3 ,],

                                                                                                                                                    [percentages_mat2_gen1, percentages_mat2_gen2 ,percentages_mat2_gen2 ,],
                                                                                                                                                                    ....
                                                                                                                                                    [percentages_mat3_gen_1 , percentages_mat3_gen_max_epoch ,percentages_mat3_gen_max_epoch] 
                                                                                                                                                    ]"""
    matrix_CEM = CovMatrix(**kwargs)
    matrix_CEMir = CovMatrix(**kwargs)
    ls_matrix = [matrix_CEM, matrix_CEMir]
    ls_update_functions = [matrix_CEM.update_covariance,
                           matrix_CEMir.update_covariance_inverse_resize]

    return {**CEM_GEN(score_function, ls_matrix, ls_update_functions, **kwargs), 'list_matrix_names': ['CEM', 'CEMir']}
