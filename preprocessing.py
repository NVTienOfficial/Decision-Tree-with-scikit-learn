from sklearn.preprocessing import OrdinalEncoder

def ordinal_encoder(train, test):
    ''' Preprocess the data with Ordinal Encoder'''
    # Take columns which will be encoded
    obj = (train.dtypes == 'object')
    obj_cols = list(obj[obj].index)

    ordinal_encoder = OrdinalEncoder()

    # encode
    train[obj_cols] = ordinal_encoder.fit_transform(train[obj_cols])
    test[obj_cols] = ordinal_encoder.transform(test[obj_cols])

    return train, test