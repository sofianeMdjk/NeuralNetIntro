from keras.models import Sequential
from keras.layers import Dense
import numpy
import preporcessing as pre

# fix random seed for reproducibility
numpy.random.seed(5)


# loading dataset

dataset = numpy.loadtxt("dataset.csv", delimiter=",")
# split into input (X) and output (Y) variables
inputs = dataset[:,0:8]
target = dataset[:,8]

#preprocessing of the input

input = pre.dataRescale(inputs)

# creating the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(input, target, epochs=150, batch_size=10, verbose=2)

# evaluating the model
scores = model.evaluate(input, target)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Saving the model
model.save('model.h5')

