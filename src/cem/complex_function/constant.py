seed= 43

centroids = {
    "Rastrigin": [-5,-5],
    "Ackley": [-5,-5],
    "Sphere": [-30,-40],
    "Rosenbrock": [-40,-40],
    "Griewank": [2,2.5],
    "booth": [-10,-10],
    "Holder": [0,-5],
    "mishra_bird": [0,6],
    "distance": [0,-6]
    }

sigma = 0.2
noise_multiplier = 0.999
max_epochs = 50
pop_size = 30
elites_nb = 10

seuil_convergence = {
    "Rastrigin": 0,
    "Ackley": 0,
    "Sphere": 0,
    "Rosenbrock": 0,
    "Griewank": 0,
    "booth": 0,
    "Holder": -19.2085,
    "mishra_bird": -106.7645367,
    "distance": 0,
    }
delta_convergence = {
    "Rastrigin": 0.1,
    "Ackley": 0.1,
    "Sphere": 9,
    "Rosenbrock": 0.1,
    "Griewank": 0.1,
    "booth": 0.1,
    "Holder": 0.1,
    "mishra_bird": 0.1,
    "distance": 0.1,
}


ls_versions_CEM  = ['CEM', 'CEMi', 'CEMir','CEM+CEMi', 'CEM+CEMiR']
version_CEM = ls_versions_CEM[2]
