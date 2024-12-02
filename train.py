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
regularizer = tf.keras.regularizers.l2(3)#0.0001)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.00001,
    amsgrad=True
)

def predictImage(model, image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (30, 30))
    normalized_image = resized_image / 255.0
    normalized_image = np.array(normalized_image)
    normalized_image = np.expand_dims(normalized_image, axis=0)
    prediction = model.predict(normalized_image)
    print(prediction)
    print(np.argmax(prediction))

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
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

previousOutputLayer = model.layers[len(model.layers) - 1].get_weights()
#print(previousOutputLayer)


predictImage(model, "teste.png")

#freeze backbone
model._layers.pop()
for layer in model.layers:
    layer.trainable = False

#add extra class
model.add(Dense(45, activation='softmax', kernel_regularizer=regularizer))
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

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


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
        'bike': 44
    }
    df['number'] = df['number'].replace(class_mapping)
    return df


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

#labels = np.full(756, 43)
dfTrain = readImages("train")
dfTest = readImages("test")
dfVal = readImages("val")

trainGen = DataGenerator(dfTrain, "mixedDataset/train", batch_size=8, shuffle=True)
testGen = DataGenerator(dfTest, "mixedDataset/test", batch_size=8, shuffle=True)
valGen = DataGenerator(dfVal, "mixedDataset/val", batch_size=8, shuffle=True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# @tf.function
# def train_step(inputs, labels, model, optimizer):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#         loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, predictions))

#     grads = tape.gradient(loss, model.trainable_variables)

#     # Freeze all neurons except for the last 2 in the output layer (Layer index 1)
#     for i, grad in enumerate(grads):
#         if i == 1:  # This corresponds to the weights of the output layer (layer 1)
#             # We freeze all neurons except the last two
#             # Get the number of neurons in the output layer
#             num_neurons = 45

#             # Create a mask to freeze neurons except the last 2
#             mask = tf.concat([tf.zeros([grad.shape[0], num_neurons - 2]), tf.ones([grad.shape[0], 2])], axis=1)
            
#             # Apply the mask to the gradients
#             grad *= mask  # This will zero out all columns except the last two


# # Training loop
# def train_model(model, train_generator, optimizer, epochs=1):
#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
        
#         # Iterate over batches in the generator
#         for batch_idx, (inputs, labels) in enumerate(train_generator):
#             train_step(inputs, labels, model, optimizer)
            
#             if batch_idx % 100 == 0:  # Print a message every 100 batches
#                 print(f"Batch {batch_idx}/{len(train_generator)}")

# # Optimizer
# optimizer = tf.keras.optimizers.Adam()

# # Train the model
# train_model(model, trainGen, optimizer, epochs=3)

history = model.fit(
    trainGen,
    epochs=13, #13,
    validation_data=valGen
)

testLoss, testAcc = model.evaluate(testGen)

print("Test loss = " + str(testLoss))
print("Accuracy = " + str(testAcc))


predictImage(model, "teste.png")
predictImage(model, "mixedDataset/val/bike/WVOWANM00YXT.jpg")


tf.keras.models.save_model(model, "new_model_3.h5")

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

predictImage(model, "teste.png")
predictImage(model, "mixedDataset/val/bike/WVOWANM00YXT.jpg")

#model.layers[len(model.layers) - 1].set_weights(currentOutputLayer)
# predictImage(model, "teste.png")
# predictImage(model, "mixedDataset/val/bike/WVOWANM00YXT.jpg")

tf.keras.models.save_model(model, "new_model_4.h5")
