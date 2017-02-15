from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import tensorflow as tf
import stl10_input

tf.python.control_flow_ops = tf

img_width, img_height = 96, 96
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

nb_train_samples = 5000
nb_validation_samples = 8000
nb_epoch = 50
nb_classes = 10

(X_train, y_train), (X_test, y_test) = stl10_input.load_data()
Y_train = np_utils.to_categorical(y_train - 1, nb_classes)
Y_test = np_utils.to_categorical(y_test - 1, nb_classes)

last = base_model.output
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(10, activation='sigmoid')(x)

model = Model(base_model.input, pred)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-5, momentum=0.9),
              metrics=['accuracy'])

model.summary()
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(X_test, Y_test, batch_size=32)

model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)
