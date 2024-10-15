##Data Preprocessing
##Import Libraries
##Load Data
##Show Image from Numbers
##Change Dimension / Feature Scaling

## Import Libraries

 

 
#Train Convolutional Neural Network on 60,000 Fashion-MNIST Images (data in NP array)
 
#Test Convolutional Neural Network on 10,000 Fashion-MNIST Images (data in NP array)
 
## Import Libraries

 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
import keras # to build Neural Network

"""## Load Data"""
 
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data() # load dataset from  keras


# Print shape of Data
 
X_train.shape, y_train.shape,  X_test.shape, y_test.shape


X_train[0] # image data in 2d numpy array shape 28x28 pixel
 
y_train[0] #9 => Ankle boot

class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


'''
0 => T-shirt/top 
1 => Trouser 
2 => Pullover 
3 => Dress 
4 => Coat 
5 => Sandal 
6 => Shirt 
7 => Sneaker 
8 => Bag 
9 => Ankle boot '''


 
"""## Show image"""
 
plt.imshow(X_train[0], cmap='Greys')
 
plt.figure(figsize=(16,16))
 
j=1
for i in np.random.randint(0, 1000, 25):
  plt.subplot(5,5,j); j+=1
  plt.imshow(X_train[i], cmap="Greys")
  plt.axis('off') # off the axis
  plt.title('{} / {}'.format(class_labels[y_train[i]], y_train[i]))
  
  
  """## Change Dimension"""
 
X_train.shape
 
X_train.ndim
 
# expected conv2d_input to have 4 dimensions, but got array with shape (28, 28, 1)
# so we have increase the dimention 3 to 4
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
 
# ref: https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html
 
X_train.ndim
 
"""## Feature Scaling"""
 
X_train = X_train/255
X_test = X_test/255
 
"""## Split Dataset"""
 
from sklearn.model_selection import train_test_split 
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size= 0.2, random_state=2020)
 
X_train.shape,  y_train.shape, X_validation.shape, y_validation.shape



"""# Convolutional Neural Network - Model Building"""
 
#Building CNN model
cnn_model = keras.models.Sequential([
                         keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding='valid',activation= 'relu', input_shape=[28,28,1]),
                         keras.layers.MaxPooling2D(pool_size=(2,2)),
                         keras.layers.Flatten(),
                         keras.layers.Dense(units=128, activation='relu'),
                         keras.layers.Dense(units=10, activation='softmax')
])
 
cnn_model.summary() # get the summary of model
 
# complie the model
cnn_model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
 
# train cnn model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=512, verbose=1, validation_data=(X_validation, y_validation))


"""# Test the Model"""
 
y_pred = cnn_model.predict(X_test)
y_pred.round(2)
 
y_test
 
cnn_model.evaluate(X_test, y_test)
 
# Visualize output
 
plt.figure(figsize=(16,16))
 
j=1
for i in np.random.randint(0, 1000,25):
  plt.subplot(5,5, j); j+=1
  plt.imshow(X_test[i].reshape(28,28), cmap = 'Greys')
  plt.title('Actual = {} / {} \nPredicted = {} / {}'.format(class_labels[y_test[i]], y_test[i], class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
  plt.axis('off')



plt.figure(figsize=(16,30))
 
j=1
for i in np.random.randint(0, 1000,60):
  plt.subplot(10,6, j); j+=1
  plt.imshow(X_test[i].reshape(28,28), cmap = 'Greys')
  plt.title('Actual = {} / {} \nPredicted = {} / {}'.format(class_labels[y_test[i]], y_test[i], class_labels[np.argmax(y_pred[i])],np.argmax(y_pred[i])))
  plt.axis('off')
 
"""## Confusion Matrix"""
 
from sklearn.metrics import confusion_matrix
 
plt.figure(figsize=(16,9))
y_pred_labels = [ np.argmax(label) for label in y_pred ]
cm = confusion_matrix(y_test, y_pred_labels)
 
# show cm 
sns.heatmap(cm, annot=True, fmt='d',xticklabels=class_labels, yticklabels=class_labels)
 
from sklearn.metrics import classification_report
cr= classification_report(y_test, y_pred_labels, target_names=class_labels)
print(cr)


"""# Save Model"""
 
cnn_model.save('fashion_mnist_cnn_model.h5') # Save model
 
# Load model
fashion_mnist_cnn_model = keras.models.load_model('fashion_mnist_cnn_model.h5')
 
Y_pred_sample = fashion_mnist_cnn_model.predict(np.expand_dims(X_test[0], axis=0)).round(2)
Y_pred_sample
 
np.argmax(Y_pred_sample[0])
 
y_test[0]


