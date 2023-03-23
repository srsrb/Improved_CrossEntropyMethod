import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from test_function import Griewank, Rosenbrock, Sphere, Ackley, Rastrigin , booth , Holder , mishra_bird
from constant import centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb
from constant import CEMi
from algorithm import CEM
import os


test_function = mishra_bird

if __name__=='__main__':
    np.random.rand(seed)
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=1, vmax = max_epochs)

    all_weights = CEM(test_function , CEMi , centroid , seed , sigma , noise_multiplier , max_epochs , pop_size , elites_nb )
    bound_lower = -10
    bound_upper = 0
    x = np.linspace(bound_lower, bound_upper, 2000)
    y = np.linspace(bound_lower, bound_upper, 2000)
    X, Y = np.meshgrid(x, y)
    Z = test_function([X, Y])
    plt.figure(figsize=(12, 12))
    plt.contourf(X, Y, Z, pop_size, cmap='viridis')
    plt.axis('square')
    ax = plt.gca()
    cmap = plt.cm.rainbow
    for generation in range(len(all_weights)):
        ax.scatter(all_weights[generation][:, 0], all_weights[generation][:, 1], color=cmap(norm(generation)), marker='o' , s=3)
        
    if CEMi:
        plt.title("CEMi evolution of "+test_function.__name__+" function for "+str(max_epochs)+" generations (centroid = " f"{centroid}) ")
    else:
        plt.title("CEM evolution of "+test_function.__name__+" function for "+str(max_epochs)+" generations (centroid = " f"{centroid}) ")
    
    if CEMi :
        name_file = "plots_"+test_function.__name__+"/plot_CEMi for "+str(max_epochs)+" generations (centroid = " f"{centroid})"
    else:
        name_file = "plots_"+test_function.__name__+"/plot_CEM for "+str(max_epochs)+" generations (centroid = " f"{centroid})"
    if not os.path.exists('./plots_'+test_function.__name__):
        os.makedirs('plots_'+test_function.__name__)
    plt.savefig(name_file+".png")
    plt.show()