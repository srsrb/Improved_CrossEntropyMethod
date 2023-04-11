import numpy as np
import matplotlib.pyplot as plt
import os


class convergence_Analyzer():
    """Analyzer
    --
      Tool to visualize  and analyze the evolution of a population of points through graphs."""

    def __init__(self, **kwargs):
        """
        Initialisation of the Analyzer


        Parameters 
        --
        score_function
                        The score function used for the execution

        all_labels

        all_percentages

        all_weights
                        List of all the weights over generations

        all_covs
                        List of all covariance matrices

        all_centroids
                        List of all centroids over generations

        folder='analyses'
                        The folder in which each new plot will be saved

        file_name="_"
                        The prefix to the name of the file that will be generated

        Returns
        ---
        An Analyzer Object

        """
        def f(x): return kwargs[x]
        self.N_matrices = len(f('all_weights'))
        self.file_name = f('file_name')
        self.ls_percentages = f('all_percentages')
        self.folder = f('folder')
        self.f = f('score_function')
        self.all_w = f('all_weights')
        self.ls_labels = f('all_labels')

        self.all_cvs = f('all_covs')
        self.all_ctrds = f('all_centroids')

    def _save_plot(self, name_of_graph: str, show: bool):
        path = f"{self.folder}/{name_of_graph}/{self.f.__name__}/"
        if not os.path.exists("./"+path):
            os.makedirs(path)
        plt.savefig(path+self.file_name+".png")
        if show:
            plt.show()

    def comparison_slope_vs_ellipsoid_axis(self, show=False) -> None:
        """Plots consecutive projections of the main axis vectors on the slope of the centroid curve :
        If gamma and sigma are the vectors of the two main axis of the centroid, and v the slope vectors, we plot the normalized curve $\dfrac {\vec{gamma} \dot \vec{v}} {||vec{\gamma}||||\vec{v}|| =  \cos(vec{\gamma},\vec{v})}$
        to see the evolution of the projection on the curve
        Parameters
        ----------
        show : bool, optional
                        Open a new Matplotlib graphic window to show the plot, by default True
        """
        def make_tangents_list():
            n = len(self.all_ctrds)
            assert n >= 2, "Not enough points to calculate the derivative in centroid_list : " + \
                str(self.all_ctrds)
            all_tangents = []
            for i in range(n-1):
                v = self.all_ctrds[i+1]-self.all_ctrds[i]
                all_tangents.append(v)
            return np.asarray(all_tangents)

        def make_ellips_main_axis_list(ls_covs):
            """
            Returns
            -------
                            list of the two eig vectors of ov matrix( main axis of ellipsoids) in ascending order of the correspnding eig values  [(sig,gamma) in centroids...]
            """
            ls_eig_v = []
            for cov in ls_covs:
                w, v = np.linalg.eigh(cov)
                print(w)
                ls_vectors = np.asarray([v[:, i] for i in range(len(w))])
                print("EIG VALUES", w)
                ls_eig_v.append(ls_vectors)
            return np.asarray(ls_eig_v)

        def get_angle_pairwise(X, Y):
            def angle(x, y): return (180/np.pi)*np.arccos(np.abs( np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))))

            return np.asarray([angle(x, y) for (x, y) in zip(X, Y)])

        V = make_tangents_list()
        ls_main_axis_matrices = [make_ellips_main_axis_list(
            ls_covs[1:]) for ls_covs in self.all_cvs]

        projSIG, projGAM = [], []
        for ls_main_axis in ls_main_axis_matrices:
            SIG, GAM = ls_main_axis[:, 0], ls_main_axis[:, 1]
            projSIG.append(get_angle_pairwise(SIG, V))
            projGAM.append(get_angle_pairwise(GAM, V))

        fig, (ax1, ax2) = plt.subplots(2, layout='constrained')
        fig.suptitle(
            f"Evolution of the projections of the main axis vectors on the slope of the path -\n{self.file_name}")
        ax1.set_title(r"$\vec{\sigma} small axis$")
        ax2.set_title(r"$\vec{\gamma} big axis$")

        for i in range(self.N_matrices):
            angles_matrice_i_SIG, angles_matrice_i_GAM = projSIG[i] = projSIG[i], projGAM[i],
            ax1.plot(angles_matrice_i_SIG,
                     label=r'$ \hat{(\vec{\sigma},\vec{v})}$')
            ax2.plot(angles_matrice_i_GAM,
                     label=r'$ \hat{(\vec{\gamma},\vec{v})}$')
        ax1.grid()
        ax2.grid()
        ax1.set_xlabel('Generation')
        ax2.set_xlabel('Generation')
        ax1.set_ylabel('Angle')
        ax2.set_ylabel('Angle')
        ax1.legend()
        ax2.legend()

        self._save_plot("slope_vs_axis", show)

    def xy_centroid_evolution(self, show=False):
        """
        Plots the evolution of the successive centroids of the population, along with the corresponding scores

        Parameters
        --
        show=false
                        open a new matplotlib window at each generation

        Side-effects
        --
        Generates a plot
        """

        fig, (ax1, ax2) = plt.subplots(2, layout='constrained')
        #fig.set_size_inches(12, 12)
        fig.suptitle(
            f"Evolution of the weights in the phase space  function\n{self.file_name}")
        ax1.set_title(
            "Evolution of the x,y coordinates of the centroids over generations")
        ax1.plot(self.all_ctrds[:, 0], label='x')
        ax1.plot(self.all_ctrds[:, 1], label='y')
        ax1.grid()
        ax1.set_xlabel('Generation')
        ax1.legend()

        ax2.set_title(
            "Evolution of the consecutive scores of the centroids over generations")

        ax2.plot([self.f(c) for c in self.all_ctrds], label='Scores')
        ax2.grid()
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Scores')

        self._save_plot("xy_centroid_evolution", show)

    def variance_evolution(self, show=False):
        """Plots the evolution of the Variance and Covariance of the dataset along the X and Y axis

        Parameters
        --
        show=false
                        open a new matplotlib window at each generation

        Side-effects
        --
        Generates a plot"""

        fig, axs = plt.subplots(ncols=self.N_matrices)
        fig.suptitle(
            f'Evolution the Variance and Covariance Estimates\n {self.file_name}')

        for k in range(self.N_matrices):
            if self.N_matrices == 1:
                ax = axs
            else:
                ax = axs[k]
            label = self.ls_labels[k]
            all_cvs = self.all_cvs[k]

            var_X = all_cvs[:, 0, 0]
            var_Y = all_cvs[:, 1, 1]
            cov_XY = all_cvs[:, 0, 1]
            ax.plot(var_Y, label=r'$\mathrm{Var}(Y)$')
            ax.plot(var_X, label=r'$\mathrm{Var}(X)$')
            ax.plot(cov_XY, label=r'$\mathrm{Cov}(X,Y)$')
            ax.set_title(label)
            ax.legend()
            ax.grid()
            ax.set_xlabel('Generation')
            ax.set_ylabel('Variance')

        self._save_plot("variance_evolution", show)

    def percentage_evolution(self, show=False):
        """
        Parameters
        --
        show=false
                        open a new matplotlib window at each generation

        Side-effects
        --
        Generates a plot if `show==True`"""

        fig, ax = plt.subplots()
        #fig.set_size_inches(12, 12)
        for i in range(len(self.ls_percentages)):
            ls_percentage = self.ls_percentages[i]
            ax.plot(ls_percentage, label=self.ls_labels[i])
        ax.set_title(f'Percentage of CEM and CEMir\n {self.file_name}')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Generation')
        ax.set_ylabel('Percentage')

        self._save_plot("percentage_evolution", show)
