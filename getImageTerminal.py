import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#from keras.preprocessing.image.load_img import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import pickle
import tensorflow as tf


def split(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    letter_image_regions = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (w >= 1 and w <= 256) and (h >= 1 and h <= 256):
            letter_image_regions.append((x, y, w, h))

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    letters = []

    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        letters.append(letter_image)

    fig, axs = plt.subplots(1,len(letters), figsize=(15,5))
   
    i = 0

    for idx, ax in enumerate(axs):
        ax.set_title(idx)
        ax.axis('off')
        ax.imshow(letters[idx], cmap='gray') 
        status = cv2.imwrite(('SeperatedImages/image%d.png'%(i)),letters[idx]) 
        print("Image written to file-system : ",status)
        print(i)  
        i += 1





train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory = 'Training',
    target_size = (32,32),
    batch_size = 32,
    class_mode = 'categorical'

)

test_generator = test_datagen.flow_from_directory(
    directory = 'Testing',
    target_size = (32,32),
    batch_size = 32,
    class_mode = 'categorical'

)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (32,32,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 26, activation = 'softmax'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()


model.fit(train_generator,
        steps_per_epoch = 20,
        epochs = 60,
        validation_data = test_generator,
        validation_steps = 16)

def get_result(result):
    if result[0][0] == 1:
        return('a')
    elif result[0][1] == 1:
        return ('b')
    elif result[0][2] == 1:
        return ('c')
    elif result[0][3] == 1:
        return ('d')
    elif result[0][4] == 1:
        return ('e')
    elif result[0][5] == 1:
        return ('f')
    elif result[0][6] == 1:
        return ('g')
    elif result[0][7] == 1:
        return ('h')
    elif result[0][8] == 1:
        return ('i')
    elif result[0][9] == 1:
        return ('j')
    elif result[0][10] == 1:
        return ('k')
    elif result[0][11] == 1:
        return ('l')
    elif result[0][12] == 1:
        return ('m')
    elif result[0][13] == 1:
        return ('n')
    elif result[0][14] == 1:
        return ('o')
    elif result[0][15] == 1:
        return ('p')
    elif result[0][16] == 1:
        return ('q')
    elif result[0][17] == 1:
        return ('r')
    elif result[0][18] == 1:
        return ('s')
    elif result[0][19] == 1:
        return ('t')
    elif result[0][20] == 1:
        return ('u')
    elif result[0][21] == 1:
        return ('v')
    elif result[0][22] == 1:
        return ('w')
    elif result[0][23] == 1:
        return ('x')
    elif result[0][24] == 1:
        return ('y')
    elif result[0][25] == 1:
        return ('z')
    else:
        return(' ')




fileName = input('Please enter your file name: ')
print(fileName)

file_name = os.path.basename(fileName)
print(file_name)

image = cv2.imread(str(file_name))
split(image) 


endResult = ''
filename = r'SeperatedImages'
for img in os.listdir(filename):
    image = tf.keras.utils.load_img('SeperatedImages/%s'%(img), target_size = (32,32))
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    #probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    result = model.predict(image)
    result = get_result(result)
    endResult += result
    
print ('Predicted Alphabet is:')
print (endResult)