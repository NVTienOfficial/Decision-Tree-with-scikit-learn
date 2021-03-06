from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

from model import decision_tree_model
from preprocessing import ordinal_encoder

def train_different_proportion(feature, label):
    for p in [[40, 60], [60, 40], [80, 20], [90, 10]]:
        # Split dataset
        feature_train, feature_test, label_train, label_test = train_test_split(feature, label,
                                                                                train_size=p[0]/100, test_size=p[1]/100,
                                                                                random_state=42)
        
        # preprocessing dataset
        feature_train, feature_test = ordinal_encoder(feature_train, feature_test)

        # train model
        model = decision_tree_model(feature_train, label_train)

        # make prediction
        preds = model.predict(feature_test)

        # get validation
        validation = classification_report(label_test, preds, zero_division=0)
        accuracy = accuracy_score(label_test, preds)
        print("+    Train/test = ", f"{p[0]}/{p[1]}:")
        print("Accuracy: ", accuracy)
        print(validation)

        # get confuse matrix
        matrix = confusion_matrix(label_test, preds)
        plt.figure(figsize=(12,12))
        sns.heatmap(matrix,annot=True,fmt=".3f",linewidths=.5,square=True,cmap='gray')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted label')
        title = f"train/test = {p[0]}/{p[1]} -> Accuracy = {accuracy}"
        plt.title(title,size=15)
        plt.show()

        # export graph
        filename = "output/proportion/tree_" + str(p[0]) + "_" + str(p[1]) 
        export_graph(model, filename)


def train_max_depth(feature, label):
    # split dataset into 80/20
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label,
                                                                            train_size=0.8, test_size=0.2, 
                                                                            random_state=10)

    # preprocessing dataset
    feature_train, feature_test = ordinal_encoder(feature_train, feature_test)

    for d in [None, 2, 3, 4, 5, 6, 7]:
        # train model
        model = decision_tree_model(feature_train, label_train, d)

        # make prediction
        preds = model.predict(feature_test)

        # get validation
        accuracy = accuracy_score(label_test, preds)
        print("+    Max depth: ", d)
        print("Accuracy:", accuracy, '\n')

        # export graph
        filename = "output/max_depth/tree_depth" + str(d)
        export_graph(model, filename)

def export_graph(model, filename):
    filename += ".dot"
    tree.export_graphviz(model, out_file=filename)