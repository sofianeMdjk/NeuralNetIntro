from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer


def dataRescale(input):
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledData = scaler.fit_transform(input)
    return rescaledData

def dataStandarization(input):
    scaler = StandardScaler().fit(input)
    rescaledData = scaler.transform(input)
    return rescaledData

def dataNormalization(input):
    scaler = Normalizer().fit(input)
    normalizedData = scaler.transform(input)
    return normalizedData

def dataBinary(input):
    binarizer = Binarizer(threshold=0.0).fit(input)
    binaryData = binarizer.transform(input)
    return binaryData