from sklearn.metrics import accuracy_score, f1_score
from tfxtend.metrics import confusion_error_matrix


def calc_metrics(y_true, y_pred, labels=["stay", "walk", "jog", "skip", "stUp", "stDown"]):
    accuracy = accuracy_score(y_true, y_pred)
    print("acc: {}%".format(accuracy * 100))
    f_measure = f1_score(y_true, y_pred, average='macro')
    print("f1 : {}%".format(f_measure * 100))

    print("confusion matrix")
    cf = confusion_error_matrix(y_pred, y_true, target_names=labels)
    print(cf)
    print("")