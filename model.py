from sklearn import tree

def decision_tree_model(feature, label, max_depth=None):
    model = tree.DecisionTreeClassifier(max_depth=max_depth)
    model.fit(feature, label)
    return model