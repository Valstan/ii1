# Модуль Keras1
import numpy as np
import theano
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


np.random.seed(123)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Тренировочный набор данных', X_train.shape)
print('Метки тренировочного набора данных', y_train.shape)
print('Тестовый набор данных', X_test.shape)
print('Метки тестового набора данных', y_test.shape)

plt.imshow(X_train[1])
plt.show()
plt.figure()
plt.imshow(X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()
plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Метки тренировочного набора данных после преобразования', y_train.shape)
print('Метки тестового набора данных после преобразования', y_test.shape)

y0 = y_train[0]
y1 = y_train[1]
y2 = y_train[2]
print(y0, y1, y2)

# Создание модели
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# Второй сверточный слой
model.add(Conv2D(32, kernel_size=3, activation='relu'))
# Создаем вектор для полносвязной сети.
model.add(Flatten())
# Создадим однослойный персептрон
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
print(hist.history)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
# Построение графика точности предсказания
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()
# Построение графика потерь (ошибок)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
# Построение графика точности предсказания
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()
# Построение графика потерь (ошибок)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()

# Запись обученной модели сети в файл my_model.h5
model.save('my_model.h5')
# Удаление модели
del model
# Загрузка обученной модели сети из файла
model = load_model('my_model.h5')
# Загрузка обученной модели сети из файла
model_New = load_model('my_model.h5')
y_train_pr = model_New.predict_classes(X_train[:3], verbose=0)
print('Первые 3 символа: ', y_train[:3])
print('Первые 3 предсказания: ', y_train_pr[:3])
