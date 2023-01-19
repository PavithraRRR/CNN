import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16

num_classes=3
IMAGE_SHAPE = [224, 224] 
batch_size=32
epochs = 5

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = (224,224,3), weights = 'imagenet', include_top = False) 

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(128, activation = 'relu')(x) 
x = Dense(64, activation = 'relu')(x) 
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator
trdata = ImageDataGenerator()
train_data_gen = trdata.flow_from_directory(directory="Train",target_size=(224,224),shuffle=False, class_mode='categorical')
tsdata = ImageDataGenerator()
test_data_gen = tsdata.flow_from_directory(directory="Test", target_size=(224,224),shuffle=False, class_mode='categorical')



#Training based on epochs 5
training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
validation_steps_per_epoch = np.ceil(test_data_gen.samples / batch_size)
    
model.fit_generator(train_data_gen, steps_per_epoch=training_steps_per_epoch, validation_data=test_data_gen, validation_steps=validation_steps_per_epoch,
                        epochs=epochs, verbose=1)
print('Training Completed!')

#Checking Accuracy by R1 Score
Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)
import sklearn.metrics as metrics
val_trues =test_data_gen.classes
from sklearn.metrics import classification_report
print(classification_report(val_trues, val_preds))

#Confustion matrix
Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
val_preds = np.argmax(Y_pred, axis=1)
val_trues =test_data_gen.classes
cm = metrics.confusion_matrix(val_trues, val_preds)
cm


#Predict new image with trained model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
img_path = 'fresh.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds=model.predict(x)
# create a list containing the class labels
class_labels = ['Apple','Banana','Orange']
# find the index of the class with maximum score
pred = np.argmax(preds, axis=-1)
# print the label of the class with maximum score
print(class_labels[pred[0]])

model.save('model.h5')
