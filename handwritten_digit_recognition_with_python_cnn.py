import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0], cmap='gray')
plt.show()
print(y_train[0])

# checking out the shape of the image in the dataset
print("Shape of X_train : {}".format(X_train.shape))
print("Shape of y_train : {}".format(y_train.shape))
print("Shape of X_train : {}".format(X_test.shape))
print("Shape of y_train : {}".format(y_test.shape))

#the last 1 signifies that the image is greyscale
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

print("Shape of X_train : {}".format(X_train.shape))
print("Shape of y_train : {}".format(y_train.shape))
print("Shape of X_train : {}".format(X_test.shape))
print("Shape of y_train : {}".format(y_test.shape))

#hot encoding the process
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""#Building the model
Sequential model allow us to build the model layer by layer.



add() function is used to adding successive layers
"""

#Declaring the model
model = Sequential()

#Declaring the layers
layer_1 = Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (28,28,1))
layer_2 = Conv2D(64, kernel_size=(3,3), activation='relu')
layer_3 = Flatten()# serves as a connection between convolution and dense layers
#activation function softmax makes the output sum up to 1 so that the output contains a series of probabilities
layer_4 = Dense(10, activation= 'softmax')

#Add the layers to the model
model.add(layer_1)
model.add(layer_2)
model.add(layer_3)
model.add(layer_4)

"""#Compiling the model
it take 3 parameters.

1. Optimizer : it controls the learning rate
2. Loss Function : we using categorical_crossentropy loss function
3. Metrics : To make things easier to interpret
"""

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

"""#Training The model"""

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 10)

"""#Predicting and Testing on Current Dataset"""

test = X_train[9]
prediction  = model.predict(test.reshape(1,28,28,1))
#It prints the ‘softmaxed’ list output consisting of 10 probabilities of the digit fed as input.The highest probability will correspond the predicted digit.
print("Prediction (Softmax) from the neural network : \n\n {}".format(prediction))

hard_maxed_prediction = np.zeros(prediction.shape)
hard_maxed_prediction[0][np.argmax(prediction)] = 1
#I converted that “softmaxed” list in form where I replaced all the elements with 0 expect the highest probability, which I replaced with 1.
print("\n\n Hard-maxed form of the prediction : \n\n {}".format(hard_maxed_prediction))

print("\n\n--------Prediction---------\n\n")
plt.imshow(test.reshape(28,28), cmap = "gray")
# It displays the test image and predicted digit corresponding to it.
plt.show()
print("\n\n Final Output: {}".format(np.argmax(prediction)))

"""#Preprocessing the real life image
Here comes the use of These are the steps for preprocessing the image:


*   Convert that image to greyscale
*   Binarize(threshold) the greyscaled image in such a way that only the digits in the image are white and rest is black
*   Using the binarized image, find contours in the image. Here, contours will provide us the individual digits in the image
*   Now, we have the digits. But we have to modify it further in such a way that it becomes a lot more similar to the images present in the training dataset.
*   Now, looking at an image in dataset. We can infer that the image has to be of shape (28, 28), it should contain the digit white colored and background black colored, and the digit in the image is not stretched to the boundaries, instead, around the digit, in each of the four sides, there is a 5 pixel region (padding) of black color. (You’’ll understand this fully if you check out any of the image from the dataset).
*   So, now for modifying our image, we’ll resize it to (18,18)
*   Then, we will add a padding of zeros (black color) of 5 pixels in each direction (top, bottom, left, right).
*   So, the final padded image will be of the size (5+18+5, 5+18+5) = (28, 28), which is what we wanted.
"""

def predict_image(path):
  image = cv2.imread(path)
  grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

  ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)

  contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  preprocessed_digits = []
  for c in contours :
    x, y, w, h = cv2.boundingRect(c)
    
    #creating a rectangle around the digit in the original image 
    #for displaying the digits fetched via contours
    cv2.rectangle(image, (x,y), (x+w, y+h), color = (16,222, 90), thickness = 3)
    
    #cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y+h , x:x+w]

    #resizing the digit to 18,18
    resized_digiit = cv2.resize(digit, (18,18))

    #padding the digits with 5 pixels of black color in each side
    padded_digit = np.pad(resized_digiit, ((5,5), (5,5)), "constant", constant_values = 0)

    #Adding the preprocessed digit to the list of preprocessed digits
    preprocessed_digits.append(padded_digit)


  print("\n\n\n------------Contoured Image ------------")
  plt.imshow(image, cmap = 'gray')
  plt.show()

  inp = np.array(preprocessed_digits)

  for digit in preprocessed_digits:

    prediction = model.predict(digit.reshape(1,28,28,1))

    print("\n\n---------------------------------------\n\n")
    plt.imshow(digit.reshape(28,28), cmap = 'gray')
    plt.show()
    print('\n\n Final Output : {}'.format(np.argmax(prediction)))
    hard_maxed_prediction = np.zeros(prediction.shape)
    hard_maxed_prediction[0][np.argmax(prediction)] = 1
    print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    print ("\n\n---------------------------------------\n\n")

predict_image('test.jpg')