from keras.models import Model
from keras.applications import ResNet50

# ResNet50
res_model = ResNet50()
# res_model.summary()

# transfert learning
conv_output = res_model.get_layer("conv5_block3_out").output # on récupère la sortie de la dernière couche convolutive
pred_output = res_model.get_layer("predictions").output

# on a deux output : la prédiction et la sortie de la dernière couche conv
model = Model(res_model.input, outputs=[conv_output, pred_output])

model.save('resnet50-vizu')
