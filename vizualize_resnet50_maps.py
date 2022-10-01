import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from scipy.ndimage import zoom
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def printPredictions(preds, numberOfClasses = 3):
    decoded = decode_predictions(preds, top=numberOfClasses)
    for pred in decoded[0]:
        print(pred[1], ': ', round(pred[2]*100, 2), '%')
    return '{firstName} {firstPred}% / {secondName} {secondPred}% '.format(firstName=decoded[0][0][1],firstPred=round(decoded[0][0][2]*100,2), secondName=decoded[0][1][1], secondPred=round(decoded[0][1][2]*100,2))

def importAndPreprocess(img_path):
    # import image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X = np.expand_dims(img, axis=0).astype(np.float32)
    return preprocess_input(X), img

# import de notre modèle
model = load_model('./resnet50-vizu')

################
images = ['cat-pinguin-1.jpg','cat-pinguin-2.jpg', 'chat0.jpg', 'cheval-zebre.jpeg',  'elephant-butterfly.png',  'elephant-duck.jpeg',  'hedgehog-eagle.jpeg',  'tutle-sandwitch.jpg', 'mug-telephone.jpg', 'object-collection.jpg']
imgName = images[9]
################

X, img = importAndPreprocess('./images/'+ imgName)

# la fonction predict nous retourne deux outputs (définit dans resnet50.py):
# - la dernière couche convolutive
# - la prédiction
conv, preds = model.predict(X)

figtitle = printPredictions(preds)

# On affiche l'ensemble des filters conv séparément, sans discrimination
scale = 224 / 7
plt.figure(figsize=(16, 16))
plt.suptitle(figtitle, fontsize=16)
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(img)
    # Zoom permet d'agrandir la taille des features maps pour obtenir les mêmes dimensions
    # que l'image originale
    plt.imshow(zoom(conv[0, :,:,i], zoom=(scale, scale)), cmap='jet', alpha=0.3)

plt.savefig('./results/feature-maps-{imgName}.png'.format(imgName=imgName))

# on peut maintenant combiner tous les filtres et surtout les multiplier par les poids
# afin de donner plus d'importance aux features maps qui ont le plus d'impact sur
# le résultat final

# index de la classe prédite (celle dont la prédiction est la plus haute)
target = np.argmax(preds, axis=1).squeeze() # squeeze pour obtenir juste l'index (int)

# on récupère les poids ENTRAINABLES de la couche de prédiction
# on n'utilisera pas les poids non-entrainables
trainable_weights, non_trainable_weights = model.get_layer("predictions").weights

# les poids responsables de l'output target se trouvent dans la matrice 
# des poids entrainables, à l'index de notre classe prédite (target).
# On converti le tenseur de poids en array numpy
weights = trainable_weights[:, target].numpy()

# on récupère le vecteur de matrice ([0]) pour passer 
# d'une dimension (1, 7, 7, 2048) à (7, 7, 2048)
# on multiplie ces deux matrices grâce à l'opérateur @
# afin de donner plus de couleur (vers le rouge) aux 
# feature map qui ont les poids les plus élevés
heatmap = conv[0] @ weights

plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.imshow(zoom(heatmap, zoom=(scale, scale)), cmap='jet', alpha=0.5)
plt.suptitle(figtitle, fontsize=16)

plt.savefig('./results/combined-maps-{imgName}.png'.format(imgName=imgName))