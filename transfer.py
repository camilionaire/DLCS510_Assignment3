################################################################################
# Name: Camilo Schaser-Hughes
# Date: Feb 14, 2022
# Assignment #3: Transfer Learning
################################################################################


# Got the starting tensorflow tutorial from:
# https://medium.com/@kenneth.ca95/
# a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b


from turtle import shape
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import random as rand

EPOCHS = 25
BATCH = 10
LEARNING = 1e-2
TRAIN_SIZE = 90
DIMS = 224   # base line is 224... takes forever though.
sgd = tf.keras.optimizers.SGD(learning_rate=LEARNING)
(bx_train, by_train), (bx_test, by_test) = keras.datasets.mnist.load_data()


def find_train_set(x, y, t):
    tot = 0
    s = np.shape(x)[0]
    emp = np.zeros(t, dtype=int)
    newX = np.zeros((t*10, 28, 28))
    newY = np.zeros((t*10,))

    while tot < t*10:
        r = rand.randint(0, s-1)
        if emp[y[r]] < t:
            newX[tot] = x[r]
            newY[tot] = y[r]
            emp[y[r]] += 1
            tot += 1

    return newX, newY

# preprocesses data.  inputs and outputs
def preprocess_mnist_data(x, y):
    # got the code to resize the image from here:
    # https://stackoverflow.com/questions/66924232/reshaping-mnist-for-resnet50
    x = np.expand_dims(x, axis=-1)
    x = tf.image.resize(x, [DIMS, DIMS])
    x = np.repeat(x, 3, axis=-1)

    xp = keras.applications.vgg19.preprocess_input(x)
    # xp = keras.applications.resnet50.preprocess_input(x)
    yp = keras.utils.to_categorical(y, 10)
    return xp, yp


print("xtrain : ", bx_train.shape)
print("ytrain : ", by_train.shape)

x_train, y_train = find_train_set(bx_train, by_train, TRAIN_SIZE)
# for reducing the test size for testing the testing
# only taking half of the testing set for every day testing.
# x_test, y_test = find_train_set(bx_test, by_test, 200)

print("small xtrain : ", x_train.shape)
print("small ytrain : ", y_train.shape)

x_train, y_train = preprocess_mnist_data(x_train, y_train)
x_test, y_test = preprocess_mnist_data(bx_test, by_test)
bx_test, by_test = preprocess_mnist_data(bx_test, by_test)

print("new xtrain : ", x_train.shape)
print("new ytrain : ", y_train.shape)

input_t = keras.Input(shape=(DIMS, DIMS, 3))
# this cuts off dense layer 1000 and global avg pooling 2D,
# add back in 2d along with dense softmax.
# res_model = keras.applications.ResNet50(include_top=False,
#                                         weights="imagenet",
#                                         input_tensor=input_t)
og_vgg_model = keras.applications.VGG19(#include_top=False,
                                        weights="imagenet",
                                        input_tensor=input_t,
                                        )

# print(og_vgg_model.summary())

vgg_model = keras.Model(inputs=og_vgg_model.input, outputs=og_vgg_model.layers[-2].output)

# res_model.layers.pop()
# for observation
# print(model.summary())

# sets all the layers to fixed.
for layer in vgg_model.layers[:]:
    layer.trainable = False
# for verification
# for i, layer in enumerate(res_model.layers):
#     print(i, layer.name, "-", layer.trainable)

# model.add(keras.layers.Dense(10, activation='softmax'))
# print(model.summary())
# creates the new model from the old model + some stuff.
model = keras.models.Sequential([
    vgg_model,
    # keras.layers.GlobalAveragePooling2D(),
    # keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])
print(model.summary())

# # this creates a file to save our model weights
# # if... we wanted to do that kind of thing
# # check_point = keras.callbacks.ModelCheckpoint(filepath="cifar10.h5",
# #                                               montor="val_acc",
# #                                               mode="max",
# #                                               save_best_only=True,
# #                                               )

# sets loss, optimizer and metrics.
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# prints a bunch of stuff out to the screen.
print('Learning Rate: ', LEARNING)
print('Batch Size: ', BATCH)
print('Epochs: ', EPOCHS)
print(f'Training Size: {TRAIN_SIZE}x10')
print(f'Image Dimensions: {DIMS}x{DIMS}')
print('Stochastic Gradient Descent')
print('Cross Entropy Loss Function')
print('MNIST DataSet')
print('Still with global_average_pooling2d and then')
print('Dense, 10, SoftMax')

history = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCHS, verbose=1,
                    validation_data=(x_test, y_test), workers=2)  # , callbacks=[check_point])

model.summary()
# model.save("cifar10.h5")
print("Over all of the testing data: \n")
test_loss, test_acc = model.evaluate(bx_test, by_test, verbose=2)
print("test acc ", test_acc)
print("test loss ", test_loss)

################################################################################
# end of graphing portion
################################################################################
# here is the stuff that plots the first and second
# graphs.
fig, (ax0, ax1) = plt.subplots(1, 2)
fig.suptitle(f'Accuracy and Loss: MNIST {TRAIN_SIZE}x10')

# left plot
ax0.plot([1 - i for i in history.history['accuracy']], label='Train Error')
ax0.plot([1 - i for i in history.history['val_accuracy']], label='Test Error')
ax0.set_title('Error %')
ax0.grid(visible=True)
ax0.legend(loc='upper right')
# right plot
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Test Loss')
ax1.set_title('Loss Value')
ax1.grid(visible=True)
ax1.legend(loc='upper right')
plt.show()
################################################################################
# end of graphing portion
################################################################################
