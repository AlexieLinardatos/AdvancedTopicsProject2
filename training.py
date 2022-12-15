
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Activation
import tensorflow as tf
from datetime import datetime

        

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.5,
                                   zoom_range = 0.5,
                                   horizontal_flip = False)  #normally true

test_datagen = ImageDataGenerator(rescale = 1./255)




evalTestLoss = []
evalTestAcc = []
evalTrainLoss = []
evalTrainAcc = []
numUnits = []
time = []



start = datetime.now()

train_generator = train_datagen.flow_from_directory(
directory = 'Training',
target_size = (32,32),
batch_size = 100,    #try changing this
class_mode = 'categorical'

)

test_generator = test_datagen.flow_from_directory(
directory = 'Testing',
target_size = (32,32),
batch_size = 100,       # 200 most optimal? 
class_mode = 'categorical'

)
model = Sequential()


model.add(Conv2D(32, (3, 3), input_shape = (32,32,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides =1))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides =1))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides =1))

model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(units = 126, activation = 'relu'))
model.add(Dense(units = 26, activation = 'softmax'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(train_generator,
            steps_per_epoch = 500,
            epochs = 12,
            validation_data = test_generator,
            validation_steps = 16,
            shuffle=True
            )

# evalTrainLoss = history.history['accuracy']
test_loss, test_acc = model.evaluate(test_generator, verbose=2) 
print(test_loss, test_acc)
evalTestAcc.append(test_acc)
evalTestLoss.append(test_loss)

time.append(datetime.now() - start)
print(time)
model.save('trainedModel.h5')

print(evalTestAcc,evalTestLoss)
print(evalTrainLoss)
print(time)

# x = numUnits
# y1 = evalTestAcc 

# figure, axis = plt.subplots(2, 1)

# axis[0].plot(x,y1,'o')
# axis[0].set_title("Accuracy")

# y2 = evalTestLoss 

# axis[1].plot(x,y2,'o')
# axis[1].set_title("Loss")


# plt.savefig('figure.png')




# filename = 'SeperatedImages/image0.png'
# test_image = tf.keras.utils.load_img(filename, target_size = (32,32))
# test_image =  tf.keras.utils.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# result = get_result(result)
# print ('Predicted Alphabet is: {}'.format(result))

# Save the entire model to a HDF5 file.



