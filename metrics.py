from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def build_confusion_matrix_majority(true_labels, pred_labels, classes=None, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(title)
    plt.show()


def build_confusion_matrix(X_test, y_test, best_pipeline):
    # Confusion matrix
    y_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=best_pipeline.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_pipeline.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title(f"Confusion Matrix")
    plt.show()


def performance_metric(y_test, y_pred):
    # Precision, Recall, F1 Score for each class
    print("Precision (Per Class):", precision_score(y_test, y_pred, average=None))
    print("Recall (Per Class):", recall_score(y_test, y_pred, average=None))
    print("F1 Score (Per Class):", f1_score(y_test, y_pred, average=None))
