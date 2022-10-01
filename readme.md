# Computer Vision projet

`Création d’une HeatMap sur la classification d’images`

ou 

```
Class Activation Mapping (CAM)
```

[Get Heatmap from CNN ( Convolution Neural Network ), AKA CAM](https://tree.rocks/get-heatmap-from-cnn-convolution-neural-network-aka-grad-cam-222e08f57a34)

[https://www.youtube.com/watch?v=ST9NjnKKvT8](https://www.youtube.com/watch?v=ST9NjnKKvT8)

# Pourquoi est-ce intéressant ?

L’armée américaine a entrainé un réseau de neurone à différencier des jeeps de char. Ce CNN n’avait qu’une faible précision une fois déployé sur le terrain. L’ensemble des photos de Jeep ont été prise dans le désert alors que l’ensemble des celles de char ont été prise en forêt. La caractéristique qui intéressait le modèle était donc l’environement qu’on lui montrait et non le sujet. Si une heat map avait été donnée, elle aurait révélée 

[Machine learning and unintended consequences - LessWrong](https://www.lesswrong.com/posts/5o3CxyvZ2XKawRB5w/machine-learning-and-unintended-consequences)

[https://en.wikipedia.org/wiki/Edward_Fredkin](https://en.wikipedia.org/wiki/Edward_Fredkin)

Once upon a time, the US Army wanted to use neural networks to automatically detect camouflaged enemy tanks.  The researchers trained a neural net on 50 photos of camouflaged tanks in trees, and 50 photos of trees without tanks.  Using standard techniques for supervised learning, the researchers trained the neural network to a weighting that correctly loaded the training set - output "yes" for the 50 photos of camouflaged tanks, and output "no" for the 50 photos of forest.  This did not ensure, or even imply, that new examples would be classified correctly.  The neural network might have "learned" 100 special cases that would not generalize to any new problem.  Wisely, the researchers had originally taken 200 photos, 100 photos of tanks and 100 photos of trees.  They had used only 50 of each for the training set.  The researchers ran the neural network on the remaining 100 photos, and without further training the neural network classified all remaining photos correctly.  Success confirmed!  The researchers handed the finished work to the Pentagon, which soon handed it back, complaining that in their own tests the neural network did no better than chance at discriminating photos.

It turned out that in the researchers' data set, photos of camouflaged tanks had been taken on cloudy days, while photos of plain forest had been taken on sunny days.  The neural network had learned to distinguish cloudy days from sunny days, instead of distinguishing camouflaged tanks from empty forest.

[Edward Fredkin - Wikipedia](https://en.wikipedia.org/wiki/Edward_Fredkin)

# Comment est-ce fait ?

$$
\lvert \frac{\partial elephant}{\partial image} \rvert

$$

On passe sur chaque pixel et on le change un peu puis on constate si l’output (couche softmax) change beaucoup. Si c’est le cas, alors le pixel est important, sinon c’est qu’il ne l’est pas.

# Description du code

- Transfert learning

Nous allons récupéré un réseau déjà entrainé, VGG16

Pour pouvoir créer des Class Activation Maps, le network doit avoir une couche de Global Average Pooling (GAP) après la dernière couche convolutive puis une couche dense.

# Modèle n°1:

French horn VS Cornet (horn)