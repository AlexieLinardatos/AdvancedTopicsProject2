
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Activation
import tensorflow as tf
from datetime import datetime

def get_result(result):

    highest = result[0][0]
    for x in range(len(result[0])):
        if highest < result[0][x]:
            highest = result[0][x]

    if result[0][0] == highest:
        return('a')
    elif result[0][1] == highest:
        return ('b')
    elif result[0][2] == highest:
        return ('c')
    elif result[0][3] == highest:
        return ('d')
    elif result[0][4] == highest:
        return ('e')
    elif result[0][5] == highest:
        return ('f')
    elif result[0][6] == highest:
        return ('g')
    elif result[0][7] == highest:
        return ('h')
    elif result[0][8] == highest:
        return ('i')
    elif result[0][9] == highest:
        return ('j')
    elif result[0][10] == highest:
        return ('k')
    elif result[0][11] == highest:
        return ('l')
    elif result[0][12] == highest:
        return ('m')
    elif result[0][13] == highest:
        return ('n')
    elif result[0][14] == highest:
        return ('o')
    elif result[0][15] == highest:
        return ('p')
    elif result[0][16] == highest:
        return ('q')
    elif result[0][17] == highest:
        return ('r')
    elif result[0][18] == highest:
        return ('s')
    elif result[0][19] == highest:
        return ('t')
    elif result[0][20] == highest:
        return ('u')
    elif result[0][21] == highest:
        return ('v')
    elif result[0][22] == highest:
        return ('w')
    elif result[0][23] == highest:
        return ('x')
    elif result[0][24] == highest:
        return ('y')
    elif result[0][25] == highest:
        return ('z')

        

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

for x in range(1,1000,100):
    if(x>=2000):
        break
    start = datetime.now()

    train_generator = train_datagen.flow_from_directory(
    directory = 'Training',
    target_size = (32,32),
    batch_size = x,    #try changing this
    class_mode = 'categorical'

    )

    test_generator = test_datagen.flow_from_directory(
    directory = 'Testing',
    target_size = (32,32),
    batch_size = x,       # try changing this 
    class_mode = 'categorical'

    )
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (32,32,3), activation = 'relu'))
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
    evalTrainLoss = history.history['accuracy']
    test_loss, test_acc = model.evaluate(test_generator, verbose=2) 
    evalTestAcc.append(test_acc)
    evalTestLoss.append(test_loss)
    numUnits.append(x)
    time.append(datetime.now() - start)
    print(time,x)
    model.save('trainedModel.h5')

print(evalTestAcc,evalTestLoss,numUnits)
print(evalTrainLoss)
print(time)

x = numUnits
y1 = evalTestAcc 

figure, axis = plt.subplots(2, 1)

axis[0].plot(x,y1,'o')
axis[0].set_title("Accuracy")

y2 = time 

axis[1].plot(x,y2,'o')
axis[1].set_title("Time")


plt.savefig('figure.png')




# filename = 'SeperatedImages/image0.png'
# test_image = tf.keras.utils.load_img(filename, target_size = (32,32))
# test_image =  tf.keras.utils.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = model.predict(test_image)
# result = get_result(result)
# print ('Predicted Alphabet is: {}'.format(result))

# Save the entire model to a HDF5 file.



