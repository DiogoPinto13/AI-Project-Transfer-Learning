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

input_shape = (30, 30, 3)

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

model = tf.keras.models.load_model("traffic_classifier.h5")
#model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

previousOutputLayer = model.layers[len(model.layers) - 1].get_weights()
print(previousOutputLayer)

#freeze backbone
model._layers.pop()
for layer in model.layers:
    layer.trainable = False

#add extra class
model.add(Dense(45, activation='softmax'))
model.summary()


#add the weights of the previous layer
currentOutputLayer = model.layers[len(model.layers) - 1].get_weights()

# print("WEIGHTS BEFORE")
# print(previousOutputLayer[0])
# print("BISASES BEFORE")
# print(previousOutputLayer[1])

#copy the parameters from the previous output layer to the new
for i in range(len(previousOutputLayer[0])):
    for j in range(len(previousOutputLayer[0][i])):
        currentOutputLayer[0][i][j] = previousOutputLayer[0][i][j]

for i in range(len(previousOutputLayer[1])):
    currentOutputLayer[1][i] = previousOutputLayer[1][i]

# print("WEIGHTS AFTER")
# print(currentOutputLayer[0])
# print("BISASES AFTER")
# print(currentOutputLayer[1])

#update the actual output layer
print("Updating the parameters...")
#print(model.layers[len(model.layers) - 1].get_weights())
model.layers[len(model.layers) - 1].set_weights(currentOutputLayer)

#print("Checking if the parameters are actually updated...")
#print(model.layers[len(model.layers) - 1].get_weights())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

#model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

trainImageNames = [f for f in os.listdir("datasetBus/data/train/minibus") if os.path.isfile(os.path.join("datasetBus/data/train/minibus", f))]
testImageNames = [f for f in os.listdir("datasetBus/data/test/minibus") if os.path.isfile(os.path.join("datasetBus/data/test/minibus", f))]
valImageNames = [f for f in os.listdir("datasetBus/data/val/minibus") if os.path.isfile(os.path.join("datasetBus/data/val/minibus", f))]


# Create a DataFrame with the file names and a column with the number 43
dfTrain = pd.DataFrame({
    'image_name': trainImageNames,
    'number': 43
})
dfTest = pd.DataFrame({
    'image_name': testImageNames,
    'number': 43
})
dfVal = pd.DataFrame({
    'image_name': valImageNames,
    'number': 43
})

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
            y.append(row.number)
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

#labels = np.full(756, 43)
trainGen = DataGenerator(dfTrain, "datasetBus/data/train/minibus", batch_size=8, shuffle=True)
testGen = DataGenerator(dfTest, "datasetBus/data/test/minibus", batch_size=8, shuffle=True)
valGen = DataGenerator(dfVal, "datasetBus/data/val/minibus", batch_size=8, shuffle=True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

history = model.fit(
    trainGen,
    epochs=5,
    validation_data=valGen
)

testLoss, testAcc = model.evaluate(testGen)

print("Test loss = " + str(testLoss))
print("Accuracy = " + str(testAcc))


#fine tuning step
for layer in model.layers:
    layer.trainable = True

print("Fine tune process")
history = model.fit(
    trainGen,
    epochs=5,
    validation_data=valGen
)

testLoss, testAcc = model.evaluate(testGen)

print("Test loss = " + str(testLoss))
print("Accuracy = " + str(testAcc))