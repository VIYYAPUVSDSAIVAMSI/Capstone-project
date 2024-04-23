import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

num_classes = 5
ima_rows, ima_cols = 48, 48
batch_size = 8

train_data_dir = r'C:\Users\saiva\PycharmProjects\pythonProject1\images\images\train'
validation_data_dir = r'C:\Users\saiva\PycharmProjects\pythonProject1\images\images\validation'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, color_mode="grayscale", target_size=(ima_rows, ima_cols), class_mode='categorical', batch_size=batch_size)

validation_generator = validation_datagen.flow_from_directory(validation_data_dir, color_mode="grayscale", target_size=(ima_rows, ima_cols),class_mode='categorical', batch_size=batch_size)
print(validation_generator)
print(train_generator)
print(validation_datagen)
print(train_datagen)

model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128,kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=24282//batch_size, epochs=10, validation_data=validation_generator, validation_steps =5937//batch_size)
model.save('emotionexpressiondata.h5')