import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import shutil

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

def runModel():
    endResult = ''
    filename = r'SeperatedImages'
    for img in os.listdir(filename):
        image = tf.keras.utils.load_img('SeperatedImages/%s'%(img), target_size = (32,32))
        image = tf.keras.utils.img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        result = probability_model.predict(image)
        result = get_result(result)
        endResult += result
    
    print ('Predicted Alphabet is:')
    print (endResult)
   

def split(img):

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    print(gray)
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
    white = [255,255,255]

    #cv2.imwrite(('SeperatedImages/a.png'),letters[0])
    for idx, ax in enumerate(axs):
        ax.set_title(idx)
        ax.axis('off')
        ax.imshow(letters[idx], cmap='gray')
        letters[idx] = cv2.copyMakeBorder(letters[idx], 50, 50, 50, 50,cv2.BORDER_CONSTANT, value= white) 
        cv2.imwrite(('SeperatedImages/image%d.png'%(i)),letters[idx]) 
        i += 1
    '''
    edges = cv2.Canny(img,100,200)
    vertical_sum = np.sum(edges, axis=0)
    vertical_sum = vertical_sum != 0
    changes = np.logical_xor(vertical_sum[1:], vertical_sum[:-1])
    #change_pts = np.nonzero(changes)[0]

    x = []
    w = []

    widthCounter = 0
    for i in range(len(changes)):
        widthCounter += 1
        if (changes[i] == True):
            if (len(x) == len(w)):
                x.append(i - 1)
                widthCounter = 0
            else:
                w.append(widthCounter + x[len(x)-1] + 1)            
                
    #print(x)
    #print(w)

    horizontal_sum = np.sum(edges, axis=1)
    horizontal_sum = horizontal_sum != 0
    changes = np.logical_xor(horizontal_sum[1:], horizontal_sum[:-1])

    y = []
    h = []

    heightCounter = 0
    for i in range(len(changes)):
        heightCounter += 1
        if (changes[i] == True):
            if (len(y) == len(h)):
                y.append(i - 1)
                heightCounter = 0
            else:
                h.append(heightCounter + y[len(y)-1] + 1)            
                
    #print(y)
    #print(h)
    letters = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray)

    for i in range(len(x)):
        letter_image = []
        letter_image = gray[y[0]:h[0], x[i]:w[i]]
        letters.append(letter_image)

    fig, axs = plt.subplots(1,len(letters), figsize=(15,5))

    i = 0
    white = [255,255,255]

    #cv2.imwrite(('SeperatedImages/a.png'),letters[0])
    for idx, ax in enumerate(axs):
        ax.set_title(idx)
        ax.axis('off')
        ax.imshow(letters[idx], cmap='gray')
        letters[idx] = cv2.copyMakeBorder(letters[idx], 50, 50, 50, 50,cv2.BORDER_CONSTANT, value= white) 
        cv2.imwrite(('SeperatedImages/image%d.png'%(i)),letters[idx]) 
        i += 1
    '''

model = tf.keras.models.load_model('trainedModel.h5')

folder = r'SeperatedImages'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


fileName = input('Please enter your file name: ')
#print(fileName)

file_name = os.path.basename(fileName)
#print('Word/'+str(file_name))

image = cv2.imread('Word/'+str(file_name))
split(image)  
runModel()