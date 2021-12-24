import pandas as pd

from program import train_different_proportion, train_max_depth

def prepare_dataset(filename:str):
    X = pd.read_csv(filename)
    X.columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6',
                'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                'c1', 'c2', 'c3', 'c4', 'c5', 'c6',
                'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
                'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
                'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
                'g1', 'g2', 'g3', 'g4', 'g5', 'g6',
                'Class']

    X.dropna(axis=0, subset=['Class'], inplace=True)
    y = X.Class
    X.drop(['Class'], axis=1, inplace=True)

    return X, y

if __name__=="__main__":
    filename = "connect-4.data"
    X, y = prepare_dataset(filename)

    print("Different proportion train/test ----------------------------------------")
    train_different_proportion(X, y)

    print("Max depth for tree -----------------------------------------------------")
    train_max_depth(X, y)