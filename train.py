import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, AveragePooling2D, Rescaling, BatchNormalization, Conv2DTranspose, MaxPool2D
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import os
import keras
from PIL import Image

input_shape = (30, 30, 3)
regularizer = tf.keras.regularizers.l2(3)#0.0001)

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_pred / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)
    

class DataGenerator(Sequence):
    def __init__(self, dataframe, root, batch_size=32, shuffle=True):
        self.df = dataframe.sample(frac=1).reset_index(drop=True) if shuffle else dataframe
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        y = []
        for _, row in batch_df.iterrows():
            image_path = f'{self.root}/{row.image_name}'
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (30, 30))
            normalized_image = resized_image / 255.0
            X.append(normalized_image)
            y.append(int(row.number))
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

def predictImage(model, image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (30, 30))
    normalized_image = resized_image / 255.0
    normalized_image = np.array(normalized_image)
    normalized_image = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(normalized_image)
    print(prediction)
    print(np.argmax(prediction))

def readImages(path):
    # Specify the parent folder path
    folder_path = 'mixedDataset/' + path + '/'

    # Initialize an empty list to store file information
    data = []
    # Loop through the subfolders (e.g., minibus, sport games motorcycle racing)
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        
        # Ensure it's a directory
        if os.path.isdir(class_path):
            # Get the list of file names in the subfolder
            file_names = os.listdir(class_path)
            
            # Add file info to the data list (full file path and class label), only for .jpg files
            for file_name in file_names:
                if file_name.lower().endswith('.jpg'):  # Filter only .jpg files
                    file_path = os.path.join(class_folder, file_name)  # Include the subfolder in the path
                    if os.path.isfile(os.path.join(class_path, file_name)):
                        data.append({'image_name': file_path, 'number': class_folder})

    # Create a DataFrame with the file information
    df = pd.DataFrame(data)

    # Replace class labels with numbers
    class_mapping = {
        'minibus': 43,
        'cab': 44
    }
    df['number'] = df['number'].replace(class_mapping)
    return df

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
# model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.15))
# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(rate=0.20))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(rate=0.25))
# model.add(Dense(43, activation='softmax'))

originalAccuracyList1 = list()
originalAccuracyList2 = list()
originalAccuracyList3 = list()
newAccuracyList1 = list()
newAccuracyList2 = list()
newAccuracyList3 = list()

def train():
    model = tf.keras.models.load_model("traffic_classifier.h5")
    model.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.00001,
                    amsgrad=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])


    previousOutputLayer = model.layers[len(model.layers) - 1].get_weights()

    #freeze backbone
    model._layers.pop()
    for layer in model.layers:
        layer.trainable = False

    #add extra class
    model.add(Dense(45, activation='softmax', kernel_regularizer=regularizer))

    #add the weights of the previous layer
    currentOutputLayer = model.layers[len(model.layers) - 1].get_weights()

    #copy the parameters from the previous output layer to the new
    for i in range(len(previousOutputLayer[0])):
        for j in range(len(previousOutputLayer[0][i])):
            currentOutputLayer[0][i][j] = previousOutputLayer[0][i][j]

    for i in range(len(previousOutputLayer[1])):
        currentOutputLayer[1][i] = previousOutputLayer[1][i]

    #update the actual output layer
    model.layers[len(model.layers) - 1].set_weights(currentOutputLayer)


    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001,
        amsgrad=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'])

    #model.summary()

    previousModel = keras.models.clone_model(model)
    print("cloning...")                                                        
    #previousModel.summary()

    

    dfTrain = readImages("train")
    dfTest = readImages("test")
    dfVal = readImages("val")

    trainGen = DataGenerator(dfTrain, "mixedDataset/train", batch_size=8, shuffle=True)
    testGen = DataGenerator(dfTest, "mixedDataset/test", batch_size=8, shuffle=True)
    valGen = DataGenerator(dfVal, "mixedDataset/val", batch_size=8, shuffle=True)


    history = model.fit(
        trainGen,
        epochs=13, #13,
        validation_data=valGen
    )

    testLoss, testAcc = model.evaluate(testGen)

    print("Test loss = " + str(testLoss))
    print("Accuracy = " + str(testAcc))
    newAccuracyList1.append(testAcc)

    #fine tuning step
    for layer in model.layers:
        layer.trainable = True


            #evaluate using the original dataset
    from sklearn.metrics import accuracy_score

    # Importing the test dataset
    y_test = pd.read_csv('Test.csv')

    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data=[]

    # Retreiving the images
    with tf.device('/GPU:0'):
        for img in imgs:
            image = Image.open(img)
            image = image.resize([30, 30])
            data.append(np.array(image))

    X_test=np.array(data)
    with tf.device('/GPU:0'):
        pred = np.argmax(model.predict(X_test), axis=-1)

    #Accuracy with the test data
    from sklearn.metrics import accuracy_score
    originalAccuracy = accuracy_score(labels, pred)
    originalAccuracyList1.append(originalAccuracy)

    print("Original dataset accuracy: " + str(originalAccuracy))

    print("Fine tune process")
    history = model.fit(
        trainGen,
        epochs=5,
        validation_data=valGen
    )

    testLoss, testAcc = model.evaluate(testGen)

    print("Test loss = " + str(testLoss))
    print("Accuracy = " + str(testAcc))
    newAccuracyList2.append(testAcc)

    #evaluate using the original dataset
    from sklearn.metrics import accuracy_score

    # Importing the test dataset
    y_test = pd.read_csv('Test.csv')

    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data=[]

    # Retreiving the images
    with tf.device('/GPU:0'):
        for img in imgs:
            image = Image.open(img)
            image = image.resize([30, 30])
            data.append(np.array(image))

    X_test=np.array(data)
    with tf.device('/GPU:0'):
        pred = np.argmax(model.predict(X_test), axis=-1)

    #Accuracy with the test data
    from sklearn.metrics import accuracy_score
    originalAccuracy = accuracy_score(labels, pred)
    originalAccuracyList2.append(originalAccuracy)

    print("Original dataset accuracy: " + str(originalAccuracy))

    #keras.backend.clear_session()
    #return
    
    print("#################### DISTILLERT TIME ###########################")
    distiller = Distiller(student=model, teacher=previousModel)
    distiller.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001,
            amsgrad=True),
        metrics=['accuracy'],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=0.1,#0.0001,
        temperature=10#0.01
    )

    history = distiller.fit(trainGen, epochs=20, validation_data=valGen)
    newDatasetLoss, newDatasetAccuracy = distiller.evaluate(testGen)

    newAccuracyList3.append(newDatasetAccuracy)
    print("New dataset accuracy: " + str(newDatasetAccuracy))

    #evaluate using the original dataset
    from sklearn.metrics import accuracy_score

    # Importing the test dataset
    y_test = pd.read_csv('Test.csv')

    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data=[]

    # Retreiving the images
    with tf.device('/GPU:0'):
        for img in imgs:
            image = Image.open(img)
            image = image.resize([30, 30])
            data.append(np.array(image))

    X_test=np.array(data)
    with tf.device('/GPU:0'):
        pred = np.argmax(distiller.predict(X_test), axis=-1)

    #Accuracy with the test data
    from sklearn.metrics import accuracy_score
    originalAccuracy = accuracy_score(labels, pred)
    originalAccuracyList3.append(originalAccuracy)

    print("Original dataset accuracy: " + str(originalAccuracy))

    #distiller.save('final_distiller.keras')
    #tf.keras.models.save_model(distiller, "distiller_2.h5")
    keras.backend.clear_session()
    distiller = None
    previousModel = None
    model = None

def main():
    for i in range(30):
        train()
    print("####################")
    print("FINAL RESULTS")
    print("####################")

    print("Lists:")
    print(originalAccuracyList1)
    print(originalAccuracyList2)
    print(originalAccuracyList3)
    print(newAccuracyList1)
    print(newAccuracyList2)
    print(newAccuracyList3)
    
    print("Before fine-tuning: ")
    print("Original dataset accuracy " + str(np.mean(originalAccuracyList1)) + " with std of " + str(np.std(originalAccuracyList1)))
    print("After fine-tuning: ")
    print("Original dataset accuracy " + str(np.mean(originalAccuracyList2)) + " with std of " + str(np.std(originalAccuracyList2)))
    print("After knowledge distillation: ")
    print("Original dataset accuracy " + str(np.mean(originalAccuracyList3)) + " with std of " + str(np.std(originalAccuracyList3)))
    
    print("##################################")
    print("New accuracy: ")
    print("Before fine-tuning: ")
    print("Original dataset accuracy " + str(np.mean(newAccuracyList1)) + " with std of " + str(np.std(newAccuracyList1)))
    print("After fine-tuning: ")
    print("Original dataset accuracy " + str(np.mean(newAccuracyList2)) + " with std of " + str(np.std(newAccuracyList2)))
    print("After knowledge distillation: ")
    print("Original dataset accuracy " + str(np.mean(newAccuracyList3)) + " with std of " + str(np.std(newAccuracyList3)))
    



if __name__ == "__main__":
    main()

