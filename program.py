from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from model import decision_tree_model
from preprocessing import ordinal_encoder

def train_different_proportion(feature, label):
    for p in [[0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [0.9, 0.1]]:
        feature_train, feature_test, label_train, label_test = train_test_split(feature, label,
                                                                            train_size=p[0], test_size=p[1], random_state=0)

        feature_train, feature_test = ordinal_encoder(feature_train, feature_test)

        model = decision_tree_model(feature_train, label_train)
        preds = model.predict(feature_test)
        acc = classification_report(label_test, preds)
        print(f"{p[0]}/{p[1]}:")
        print(acc)

def train_max_depth(feature, label):
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label,
                                                                            train_size=0.8, test_size=0.2, random_state=0)

    feature_train, feature_test = ordinal_encoder(feature_train, feature_test)

    for d in [None, 2, 3, 4, 5, 6, 7]:
        model = decision_tree_model(feature_train, label_train, d)
        preds = model.predict(feature_test)
        acc = classification_report(label_test, preds)
        print("Max depth:", d)
        print(acc)