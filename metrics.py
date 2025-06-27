from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def build_confusion_matrix(X_test, y_test, best_pipeline):
    # Confusion matrix
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=best_pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_pipeline.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix")
    plt.show()


def performance_metric(y_test, y_pred):
    # Precision, Recall, F1 Score for each class
    print("Precision (Per Class):", precision_score(y_test, y_pred, average=None))
    print("Recall (Per Class):", recall_score(y_test, y_pred, average=None))
    print("F1 Score (Per Class):", f1_score(y_test, y_pred, average=None))
