import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import shutil

#this function takes an array of probabilities and returns the letter of the highest probability
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

#this function uses the model to predict the charaters 
def runModel():
    endResult = ''
    filename = r'SeperatedImages'
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()]) #create probability model based on saved model

    for img in os.listdir(filename):        #go through the directory
        image = tf.keras.utils.load_img('SeperatedImages/%s'%(img), target_size = (32,32)) #resize image to 32,32
        image = tf.keras.utils.img_to_array(image)  #conver to an array
        image = np.expand_dims(image, axis = 0)  #make sure array is proper size
        result = probability_model.predict(image)   #predict through probability model
        result = get_result(result) #get letter correlation with result
        endResult += result     #add to sting
    
    print ('Predicted Alphabet is:')
    print (endResult)   #print final string
   

#this function takes an image and splits the letters to save as idivudal character image
def split(img):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #convert to grayscale image
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)   #give boarder padding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]    #Get strong and weak threshold (for edge detection)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #find contours in image
    contours = contours[0]  #only store index 0, the location iformation

    letter_image_regions = []

    for contour in contours:    #for each contour
        (x, y, w, h) = cv2.boundingRect(contour)    #place a bounding box over it
        if (w >= 1 and w <= 256) and (h >= 1 and h <= 256):     #getting rid of all contours that do not give a full letter
            letter_image_regions.append((x, y, w, h))   #add only letters to image regions

    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])     #sort based of order of x

    letters = []

    for letter_bounding_box in letter_image_regions: #for each letter found
        x, y, w, h = letter_bounding_box    #record the bounding box 
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]   #copy the gray image based on boudning box
        letters.append(letter_image)    #add to letters

    fig, axs = plt.subplots(1,len(letters), figsize=(15,5))
   
    i = 0
    white = [255,255,255]

    #cv2.imwrite(('SeperatedImages/a.png'),letters[0])
    for idx, ax in enumerate(axs):
        letters[idx] = cv2.copyMakeBorder(letters[idx], 50, 50, 50, 50,cv2.BORDER_CONSTANT, value= white) #add whitspace boarder to split letters
        cv2.imwrite(('SeperatedImages/image%d.png'%(i)),letters[idx])   #save each letter
        i += 1
   
    #the following is the virtical split version of image splitting
    '''
    edges = cv2.Canny(img,100,200)      #using canny to record edges
    vertical_sum = np.sum(edges, axis=0)    
    vertical_sum = vertical_sum != 0    #making sure there are edges
    changes = np.logical_xor(vertical_sum[1:], vertical_sum[:-1])   #fiding the virtical changes between the images (where one letter ends and the next begins)

    x = []
    w = []

    #the following loops through the virtical cahnges and records where one letter starts and then stops
    widthCounter = 0
    for i in range(len(changes)):
        widthCounter += 1
        if (changes[i] == True):
            if (len(x) == len(w)):
                x.append(i - 1)
                widthCounter = 0
            else:
                w.append(widthCounter + x[len(x)-1] + 1)            

    #The following finds the highest and lowest edge of structure horizontally
    horizontal_sum = np.sum(edges, axis=1)
    horizontal_sum = horizontal_sum != 0
    changes = np.logical_xor(horizontal_sum[1:], horizontal_sum[:-1])

    y = []
    h = []

    loops and records what pixel and height the horzontal edge is
    heightCounter = 0
    for i in range(len(changes)):
        heightCounter += 1
        if (changes[i] == True):
            if (len(y) == len(h)):
                y.append(i - 1)
                heightCounter = 0
            else:
                h.append(heightCounter + y[len(y)-1] + 1)            
    
    letters = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #the following records the array of bounding boxes to seperate the letters
    for i in range(len(x)):
        letter_image = []
        letter_image = gray[y[0]:h[0], x[i]:w[i]]
        letters.append(letter_image)

    fig, axs = plt.subplots(1,len(letters), figsize=(15,5))

    i = 0
    white = [255,255,255]

    #write each letter to the specified directory
    for idx, ax in enumerate(axs):
        ax.set_title(idx)
        ax.axis('off')
        ax.imshow(letters[idx], cmap='gray')
        letters[idx] = cv2.copyMakeBorder(letters[idx], 50, 50, 50, 50,cv2.BORDER_CONSTANT, value= white) 
        cv2.imwrite(('SeperatedImages/image%d.png'%(i)),letters[idx]) 
        i += 1
    '''

model = tf.keras.models.load_model('trainedModel.h5')       #Load in saved nural net

folder = r'SeperatedImages'                                 #Where the split images are stored

#delete all previous images in folder
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#find image to split and run
fileName = input('Please enter your file name: ')

file_name = os.path.basename(fileName)

image = cv2.imread('Word/'+str(file_name)) #load in image
split(image)  
runModel()