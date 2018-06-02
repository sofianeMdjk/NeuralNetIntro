from keras.models import load_model

import numpy


#importing the model
model = load_model('model.h5')

#predict an input

sample = numpy.array([[1,103,30,38,83,43.3,0.183,33]])
prediction = model.predict(sample)


rounded = [round(x[0]) for x in sample]
print(rounded)