# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import pickle

inputsize = (128,128)
batch_size = 32

#loading training and testing data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training',
                                                 color_mode='grayscale',
                                                 target_size = inputsize,
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/testset',
                                            color_mode='grayscale',
                                            target_size = inputsize,
                                            batch_size = batch_size,
                                            class_mode = 'categorical')


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (*inputsize, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = training_set.num_classes, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images


classifier.fit_generator(training_set,
                         steps_per_epoch = training_set.samples/batch_size,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = test_set.samples/batch_size,
                         workers=20,
                         #use_multiprocessing=True,
                         max_queue_size=100
                         )



# save data
classifier.save('face_recog_93.h5')
with open('classes_93.pkl', 'wb') as f:
    pickle.dump(training_set.class_indices,f)
