# Reports
## February 21st 
Bonjour,
Après avoir remarqué que la CEM est plus performante que la iCEM (comme les graphes le montre), nous allons, en premier lieu, essayé d’appliquer notre code sur le notebook de Cross Enthropy methode à 2 dimensions afin de voir si tout marche bien et pour pouvoir visualiser le développement des ellipses.
Dans le cas ou tout fonctionne bien sur 2 dimensions, nous allons testé sur des environnements à plusieurs dimensions de la façon suivante :
- Extraire deux paramètres afin de les optimiser avec la CEM et iCEM.
- Appliquer la CEM sur tous les paramètres en remplaçant à chaque fois avec les deux paramètres déjà optimisé ( Dans chaque génération).
- Comparer les résultats des tests.
Et comme le 15 mars c’est le dernier délai pour rendre le rapport, nous allons commencer la rédaction de ce rapport dés cette semaine .


Condialement


Anyes, Julien, Robin


## Report of the 14 february 2023

A TRADUIRE :
> On a remarque que dans certains environnements, la CEMi fait pire
> que la methode standard et on a une proposition d'explication:
> Avec l'interpretation de la matrice de covariance en terme
> d'ellipsoide :  inverser la matrice de covariance revient a effectuer
> une rotation de pi/2 des vecteurs propres de la matrice ( qui
> correspondent aux deux directions des axes de l'ellipse)
> MAIS en faisant cela, on inverse aussi les valeurs propres de cette
> matrice, qui correspondent aux longueurs des deux axes de l'ellipse
>
> En somme, dans la CEMi, on gagne peut etre tu temps en se placant
> directement dans la direction de la meilleure evolution, mais on en
> perd parce que l'ellipse est plus petite, donc on fait de plus petits
> pas a chaque fois
> -> Solution proposee : etirer l'ellipse de CovMAtrix^{-1} de maniere
> proportionnelle  aux valeurs propres de CovMatrix. De la sorte, on
> fait de plus grands pas dans la bonne direction

### Goals for next week
- Now that we have solve the inversion issue, try some runs with complete matrix
- Solve the compute resources issue 
- Develop the creation of the json files to give to rliable
- Explore the homothety hypothesis

-------------

## Report of the 7 february 2023

We had issues with the inversion of the covariance matrix, for more information I refer you to Inversion_Issue.md

### Reunion reports
- Discussion about the inversion issue
- Try to reduce/modify the size of our neural network 

### Goals for next week
- Solve the inversion issue
- Performance comparison of the CEM and the CEMi with diagonal matrices
- Handling of rliable and its utilisation

-------------

## Report of the 31 january 2023

### Reunion reports
- Discussion about technical issues
- Comments about rliable

### Goals for next week
- Fix those technical issues
- Make the CEM work on different environments
- Modify the CEM to exploit the inverse of the covariance matrix
- Make the CEMi work on different environments
+ Generate a performance file usable by the rliable library if possible

---------------

## Report of the 19 january 2023

- Organisation of deadlines
- Intern organisation
- One reunion per week
