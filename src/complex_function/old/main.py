import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from convergence_analysis import *
from test_function import *
from constant import *
from algorithm import *
import os


score_function = Ackley


def plot_cem_on_function(f, folder_plots ="plots_2",folder_analysis ="analysis_2",
                             show_plot=False, show_analysis=False,
                             show_points=False, show_ellipsoids=False,show_centroids=True):
    """Plot an exectution of the CEM with `f` as the score function

    Parameters

    ------

    `f` 
        function that takes a vector of `R^2` as an input an returns a real number

    folder_plots ="plots_"
        The directory in which the plot will be generated and saved
    
    folder_analysis ="analysis_"
        The directory in which the analysis will be generated and saved


    show_plot=False
        Show the plot generated for the execution
    
    show_analysis=False
        Show the analysis generated for the execution
    
   
    `show_points (=False)`
        if `show_plot==True`
            if `show_points==True` then the plot draws the points corresponding to every every individuals.
    
    ` show_ellipsoids=True` `show_centroids=True`
        Draw the ellipsoids
    
    show_centroids=True
        draw the centroids at each generation

    Side-Effects

    ---

    Plots the execution (according to the parameters in the module `constant`) ans saves it  """

    np.random.rand(seed)
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1, vmax = max_epochs)


    all_weights, all_centroids, all_covs , percentage_CEM , percentage_CEMir = CEM(f,merge, EXPERIMENT, CEMi , centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb )
    
    x_min,y_min = np.min(all_weights[:,:,0]),np.min(all_weights[:,:,1])
    x_max,y_max = np.max(all_weights[:,:,0]),np.max(all_weights[:,:,1])
    
    x = np.linspace(x_min-1, x_max+1, num=200)
    y = np.linspace(y_min-1, y_max+1, num=200)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    plt.figure(figsize=(12, 12))
    plt.contourf(X, Y, Z, pop_size, cmap='viridis')
    plt.colorbar()
    plt.grid()
    ax = plt.gca()
    cmap = plt.cm.rainbow
    
    for generation in range(len(all_weights)):
        weights = all_weights[generation]
        cov     = all_covs[generation]
        if show_points:
            ax.scatter(weights[:, 0], weights[:, 1], color=cmap(norm(generation)), marker='o' , s=3)
        if show_ellipsoids:
            if merge :
                cov_2 = all_covs[2*generation + 1]
                plot_confidence_ellipse(cov, ax,weights, n_std=1., edgecolor='darkorange')
                plot_confidence_ellipse(cov_2, ax,weights, n_std=1., edgecolor='b')
            else :
                plot_confidence_ellipse(cov, ax,weights, n_std=1., edgecolor='b')  
    
    if show_centroids:
        ax.plot(all_centroids[:,0],all_centroids[:,1],marker='^',c='r')
    
    if merge:
        cem_variant  =  ' merged with CEMir'
    elif EXPERIMENT: # If we want to test the variant of the CEm that we experiment on
        cem_variant  =  'iResize' 
    elif  CEMi :
        cem_variant = 'i' 
    else:
        cem_variant =  ''
    plt_title  = f" {f.__name__} function - CEM{cem_variant} evolution - {max_epochs} generations (starting point = {centroid}) "
    plt.title(plt_title)
    file_name =  f"{f.__name__}/CEM{cem_variant}_{max_epochs}G_starting_point{centroid}) "
    if not os.path.exists(f'./{folder_plots}/{f.__name__}'):
        os.makedirs(f'{folder_plots}/{f.__name__}')
    plt.savefig(folder_plots+'/'+file_name+".png")
    if show_plot:
        plt.show()

    analyzer  =  Analyzer(f ,all_weights,all_covs,all_centroids,folder=folder_analysis, file_name=plt_title)
    analyzer.xy_centroid_evolution()
    analyzer.variance_evolution()
    if merge:
        analyzer.percentage_evolution(percentage_CEM , percentage_CEMir)
    

def plot_every_function(folder_plots = "plots_2",folder_analysis ="analysis_2" ,show_plot=False,show_analysis=False,show_points=False, show_ellipsoids=False,show_centroids=True):
    
    """Generate a plot for every function in the module `test_function` along with an Analyse for every execution
    
    Parameters
    ---
     folder_plots ="plots_"
        The directory in which the plots will be generated and saved
    
    folder_analysis ="analysis_"
        The directory in which the analysis will be generated and saved

    show_plot=False
        Show the plot generated for the execution
    
    show_analysis=False
        Show the analysis generated for the execution
    
   
    `show_points (=False)`
        if `show_plot==True`
            if `show_points==True` then the plot draws the points corresponding to every every individuals.
    
    ` show_ellipsoids=True` `show_centroids=True`
        Draw the ellipsoids
    
    show_centroids=True
        draw the centroids at each generation

    Side-Effects

    ---

    Plots the execution (according to the parameters in the module `constant`) ans saves it 

    """
    import inspect
    import test_function
    ls_funct = np.array(inspect.getmembers(test_function,inspect.isfunction))[:,1]
    print(ls_funct)
    for f in ls_funct:
        print(f)
        plot_cem_on_function(f, folder_plots =folder_plots,folder_analysis =folder_analysis,
                             show_plot=show_plot, show_analysis=show_analysis,
                             show_points=show_points, show_ellipsoids=show_ellipsoids,show_centroids=show_centroids)

if __name__=='__main__':
    #generate a plot PLUS an analyse for every function in the module `test_function` , parameterized by the module `constant ` 
    #plot_every_function(show_points=False,show_plot=False,show_ellipsoids=False,show_centroids=True)

    #only plot the cem on the function `score_function`
    plot_cem_on_function(score_function,show_points=1,show_plot=1,show_ellipsoids=1,show_centroids=1)
  




