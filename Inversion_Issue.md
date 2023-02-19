# Compte rendu prblm matrice inverse

## Probleme  
Avec les parametres par defaut,  actor_hidden_size: [8, 8], les inversions successives des matrices de cov aboutissent, vers la 3e generation, a une erreur “ Matrice non definie Positive”

## Comportement
L’inversion classique en nombre flottants est par nature instable et inexacte. Lors des inversions successives, des erreurs d’arrondi s’accumulent. Particulierement pour les matrices de grande taille.
Lors de l’appel a la fonction MultivariateNormal ( centroid, matrice_de_cov), torch effectue une verification du caractere Positif defini de la matrice de cov selon le critere suivant:

Positive_definite ⇔ ( symmetrical AND all eigenvalues are real and positive )



Or, le calcul des valeurs propres de la matrice de cov aboutit parfois a des resultats ayant une partie imaginaire tres petite ( de l’ordre de 10^ {-6})
Donc le test precedent echoue et le programme plante


5 + 0.j
## Solution 
Utiliser l’inversion d e Cholesky  :  
methode specifique aux matrices symetriques definies positives
Semble eliminer les erreurs d’arrondi -> toutes les valeurs propres sont reelles strictement positives ( enfin!)

## methode 
Calculs faits sur la matrice entiere qu’on n’a pas rogne pour garder la diagonale ( d’ou le ```if False``` a la ligne 36 de mon code )

1 - reduction de la dimension du Reseau de neurone: actor_hidden_size: [1, 1]
2 - Augmenter la taille au fur et a mesure: erreur vers [15,1]
3 -  Affichage des valeurs propres avec torch.linalg.eigvals
4  -  Application du fix avec cholesky et verification en faisant
 matrice_cov * cholesky_inverse(matrice_cov)
-> Donne la matrice identite a 10^-6 pres. 

## Remarque : 
En testant la CEM avec le paramètre [1,1], j’ai remarque un comportement bizarre:
CEM classique :   


a … creuser
