import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.optimizers import adam
import time
import os
start_time = time.time()

# Code written by Sean McGuire

# fp = np.memmap("G:\\Consult Tumors\\paddedArray.dat", mode='r+', dtype='float16', shape=(130, 545, 512, 512))
# print("Data read in: " + str(time.time() - start_time))
# trainData = fp[:91]
# trainData = trainData/128.
# train = np.memmap("G:\\Consult Tumors\\train.dat", mode='w+', dtype='float16', shape=np.shape(trainData))
# train[:] = trainData[:]
# testData = fp[91:]
# testData = testData/128.
# test = np.memmap("G:\\Consult Tumors\\test.dat", mode='w+', dtype='float16', shape=np.shape(testData))
# test[:] = testData[:]

ExperimentName = "Batch1"

trainData = np.memmap("G:\\Consult Tumors\\train.dat", mode='r+', dtype='float16', shape=(91, 545, 512, 512))
testData = np.memmap("G:\\Consult Tumors\\test.dat", mode='r+', dtype='float16', shape=(39, 545, 512, 512))
print("Preprocessing Complete: " + str(time.time() - start_time))
loadLabels = np.load("G:\\Consult Tumors\\diag.npy")
diag = np_utils.to_categorical(loadLabels)
trainLabels = diag[:91, :]
testLabels = diag[91:, :]
print("Begin Model Initialization: " + str(time.time() - start_time))
#Model
model = Sequential()
# TODO: Add batch Norm
# TODO: Add 3D convolutions
#Convolution2D(numofFilters,input_shape(x,y,z)
model.add(Conv2D(4, (2, 2), input_shape=(545, 512, 512), strides=(5, 5), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(5,5)))

model.add(Conv2D(8, (3, 3), strides=(3, 3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(5,5)))

model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

try:
    os.mkdir('G:\\Consult Tumors\\Logs\\' + ExperimentName + "\\")
except FileExistsError:
    pass
tbCallBack = TensorBoard(log_dir='G:\\Consult Tumors\\Logs\\' + ExperimentName + "\\", histogram_freq=0, batch_size=1, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
a = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=a,
              metrics=['accuracy'])
print("Model Compiled: " + str(time.time() - start_time))
print(model.summary())
Callback()
earlyStop = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=2, mode='auto')
callbacks_list = [tbCallBack,earlyStop]
print("Begin Trainning: " + str(time.time() - start_time))
model.fit(trainData, trainLabels, batch_size=1, epochs=300, validation_data=(testData, testLabels), verbose=2, callbacks=callbacks_list)

#convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
elapsed_time = time.time() - start_time
print ("Completed: " + str(elapsed_time))

model.save('my_model3conlayer.h5')

# serialize model to JSON
model_json = model.to_json()
with open("model3conlayer.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("modelWeight3conlayer.h5")
print("Saved model to disk")
