import numpy as np
import matplotlib.pyplot as plt
import os


class Analyzer():
    """Analyzer
    --
      Tool to visualize  and analyze the evolution of a population of points through graphs."""

    def __init__(self,score_function ,all_weights,all_covs,all_centroids,folder='analysis_',file_name="_"):
        """
        Initialisation of the Analyzer
        

        Parameters
        --
        score_function
            The score function used for the execution
            
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
        self.file_name  = file_name
        self.folder=folder
        self.f = score_function
        self.all_w = all_weights
        self.all_cvs = all_covs
        self.all_ctrds = all_centroids

    
    def xy_centroid_evolution(self,show=False):
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
        
        fig,(ax1,ax2) = plt.subplots(2,layout='constrained')
        #fig.set_size_inches(12, 12)
        fig.suptitle(f"Evolution of the weights in the phase space  function\n{self.file_name}")
        ax1.set_title("Evolution of the x,y coordinates of the centroids over generations")
        ax1.plot(self.all_ctrds[:,0],label='x') 
        ax1.plot(self.all_ctrds[:,1],label='y')
        ax1.grid()
        ax1.set_xlabel('Generation')
        ax1.legend()

        ax2.set_title("Evolution of the consecutive scores of the centroids over generations")
        
        ax2.plot([self.f(c) for c in self.all_ctrds],label='Scores')
        ax2.grid()
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Scores')

        path  = f"{self.folder}/xy_centroid_evolution/{self.f.__name__}/"
        if not os.path.exists("./"+path):
            os.makedirs(path)
        plt.savefig(path+self.file_name+".png")
        if show:
            plt.show()
        

    
    def variance_evolution(self,show=False ):
        """Plots the evolution of the Variance and Covariance of the dataset along the X and Y axis
        
        Parameters
        --
        show=false
            open a new matplotlib window at each generation
            
        Side-effects
        --
        Generates a plot"""

        fig,ax = plt.subplots()
        #fig.set_size_inches(12, 12)
        var_X =self.all_cvs[:,0,0]
        var_Y = self.all_cvs[:,1,1]
        cov_XY = self.all_cvs[:,0,1]
        ax.plot(var_Y,label=r'$\mathrm{Var}(Y)$') 
        ax.plot(var_X,label=r'$\mathrm{Var}(X)$') 
        ax.plot(cov_XY,label=r'$\mathrm{Cov}(X,Y)$') 
        ax.set_title(f'Evolution the Variance and Covariance Estimates\n {self.file_name}')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Generation')
        ax.set_ylabel('Variance')
        path  = f"{self.folder}/variance_evolution/{self.f.__name__}/"
        if not os.path.exists("./"+path):
            os.makedirs(path)
        plt.savefig(path+self.file_name+".png")
        
        if show:
            plt.show()
    
    def percentage_evolution(self,percentage_CEM , percentage_CEMir ,show=False):
        """
        Parameters
        --
        show=false
            open a new matplotlib window at each generation
            
        Side-effects
        --
        Generates a plot"""

        fig,ax = plt.subplots()
        #fig.set_size_inches(12, 12)
        ax.plot(percentage_CEM,label='percentage_CEM') 
        ax.plot(percentage_CEMir,label='percentage_CEMir') 
        ax.set_title(f'Percentage of CEM and CEMir\n {self.file_name}')
        ax.legend()
        ax.grid()
        ax.set_xlabel('Generation')
        ax.set_ylabel('Percentage')
        path  = f"{self.folder}/percentage_evolution/{self.f.__name__}/"
        if not os.path.exists("./"+path):
            os.makedirs(path)
        plt.savefig(path+self.file_name+".png")
        
        if show:
            plt.show()

            