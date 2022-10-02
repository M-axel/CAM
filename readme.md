# Class Activation Mapping (CAM)

## Pourquoi est-ce intéressant ?

Dans <i>What Artificial Experts Can and Cannot Do</i> (1992):

> Once upon a time, the US Army wanted to use neural networks to automatically detect camouflaged enemy tanks. The researchers trained a neural net on 50 photos of camouflaged tanks in trees, and 50 photos of trees without tanks.  Using standard techniques for supervised learning, the researchers trained the neural network to a weighting that correctly loaded the training set - output "yes" for the 50 photos of camouflaged tanks, and output "no" for the 50 photos of forest.  This did not ensure, or even imply, that new examples would be classified correctly.  The neural network might have "learned" 100 special cases that would not generalize to any new problem.  Wisely, the researchers had originally taken 200 photos, 100 photos of tanks and 100 photos of trees.  They had used only 50 of each for the training set.  The researchers ran the neural network on the remaining 100 photos, and without further training the neural network classified all remaining photos correctly.  Success confirmed!  The researchers handed the finished work to the Pentagon, which soon handed it back, complaining that in their own tests the neural network did no better than chance at discriminating photos.
> It turned out that in the researchers' data set, **photos of camouflaged tanks had been taken on cloudy days, while photos of plain forest had been taken on sunny days**. The neural network had learned to **distinguish cloudy days from sunny days**, instead of distinguishing camouflaged tanks from empty forest.

L'utilité de pouvoir visualiser ces features maps se trouve dans la validation de notre réseaux. Être sûr que la prédiction qu'il donne a été faite en regardant ce que nous aurions utiliser en temps qu'humain pour faire de même.
Après avoir construit notre réseaux, nous expérimenterons avec des photomontages d'animaux.

Ces visualisations ont aussi des vertues pédagogiques puisqu'on entend souvent dire que les réseaux de neurones sont des <i>boîtes noires</i>.

## Comment est-ce fait ?

### Explication Mathématiques
L'explication mathématiques est plus simple a comprendre que le code:

Display equation: $$\lvert \frac{\partial elephant}{\partial image} \rvert$$

On passe sur chaque pixel d'une photo d'elephant et on le change un peu puis on regarde si l’output (couche softmax) change beaucoup. Si c’est le cas, alors le pixel est important, sinon c’est qu’il ne l’est pas.

### Au niveau code
Pas de dérivée partielle ici, tout est statique.

L'ensemble des features maps se trouvent sous forment d'un array de matrice de poids (chaque matrice correspond à une feature map). Nous recupérons ces poids à la sortie de la dernière couche convolutive.

Le problème étant que les features n'ont pas toutes la même importance pour la prédiction finale.
Un fois la prédiction obtenue, on sait quels poids (entrainables) ont joués en faveur de celle-ci. Il suffit donc de multiplié les features map avec les poids correspondant afin d'avoir des features map 'pondérées'.

## Le réseau

### VGG16

J’ai essayé de récupéré le réseau VGG16, entrainé à distinguer chien et chat, que nous avions construit en TP.

Pour créer des Class Activation Maps, le network doit avoir une couche de Global Average Pooling (GAP) après la dernière couche convolutive. VGG16 n’a pas cette couche. J’ai donc récupéré VGG16 et lui ai collé cette couche GAP

Mon entrainement et mes prédictions se faisaient correctement. Malheureusement, l’architecture complexe de ce modèle a amené à des erreurs mystiques. Je l’ai donc abandonner pour passer sur un réseau plus simple et me concentrer sur le sujet.

### ResNet50

Ce réseau possède déjà une architecture qui permet de créer la visualisation des features maps très simplement. Il m’a fallu déclarer deux outputs différents lors du predict:

1. La classification
2. La sortie de la dernière couche convolutive

## Expérimentations

TODO

## Ressources

Une bonne partie de la technique provient de l'article médium:
[Get Heatmap from CNN ( Convolution Neural Network ), AKA CAM](https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34)

Vidéo [CS 152 NN—15: Visualizing CNNs: Heat Maps](https://www.youtube.com/watch?v=ST9NjnKKvT8)

Article [Class activation maps in Keras for visualizing where deep learning networks pay attention](https://jacobgil.github.io/deeplearning/class-activation-maps)

