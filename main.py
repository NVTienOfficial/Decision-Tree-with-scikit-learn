import pandas as pd
import os

from program import train_different_proportion, train_max_depth

def prepare_dataset(filename:str):
    X = pd.read_csv(filename)

    X.dropna(axis=0, subset=['Class'], inplace=True)
    y = X.Class
    X.drop(['Class'], axis=1, inplace=True)

    return X, y

if __name__=="__main__":
    os.system('cls')

    filename = "input/connect-4.data"
    X, y = prepare_dataset(filename)

    print("Different proportion train/test ----------------------------------------")
    train_different_proportion(X, y)

    print("Max depth for tree -----------------------------------------------------")
    train_max_depth(X, y)