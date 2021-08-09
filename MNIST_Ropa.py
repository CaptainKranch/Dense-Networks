import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist #imports la data de MNIST con keras
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

var = train_images.shape #Muestra cuantas imagenes hay por su tamanno (60000, 28, 28)
var1 = test_images.shape # (10000, 28, 28)
print(var, var1)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
'''
### Estamos trabajando en escalas de grises (0-255), para que la informacion no sea muy dificil de manejar
### reescalamos a (0-1) dividiendo por 255.0

train_images = train_images/255.0
train_labels = train_labels/255.0

#print(train_images, test_images)

### Creamos nuestro modelo con las capas necesarias.
### Hay muchas funciones de activacion, podemos cambiarlas
### y verificar con cual trabaja mejor nuestra red.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #input layer 1 -----> (28,28) => 28 x 28 = 784
    keras.layers.Dense(128, activation='relu'), #hiden layer 2
    keras.layers.Dense(10, activation='softmax') #output layer 3 ----> 10 es igual al numero de clases
])

### Optimizador es el que hace de gradiente, loss calcula cuanto
### se perdio de lo esperado y metrics sera el output

model.compile(optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 10)

#Evaluamos la red neuronal con .evaluate y debemos de buscar el mejor resultado posible.
# si fuera < 70 deberiamos de cambiar los hyperparametros
#como los modelos de activacion o los epochs, hasta obtener un mejor resultado

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy: ', test_acc,
        'Test loss: ', test_loss)

#Testeamos por cada elemento y ver si es lo que se esperaba, pidiendole un numero al usuario

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmpa=plt.cm.binary)
    plt.title('Expected: ' + label)
    plt.xlabel('Guess: ' + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def predict(model, image, correcrt_label):

    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    prediction = model.predict(np.array(image))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correcrt_label], predicted_class)


def get_number():
    while True:
        num = input('Escoja un numero: ')
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 10000:
                return int(num)
            else:
                print('Errorrrrrrr')


num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
