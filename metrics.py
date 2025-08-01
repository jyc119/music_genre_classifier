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


training_rnn = [0.5623854441414458, 0.7778175153113684, 0.8707586391881622, 0.9244043095355179, 0.9374580893200412, 0.9570834637221154, 0.9662032276811658, 0.9664267513076132, 0.9723724797711119, 0.9714336805400331]
validation_rnn = [0.67, 0.79, 0.795, 0.795, 0.83, 0.805, 0.795, 0.785, 0.795, 0.805]
training_cnn = [0.5623854441414458, 0.7778175153113684, 0.8707586391881622, 0.9244043095355179, 0.9374580893200412, 0.9570834637221154, 0.9662032276811658, 0.9664267513076132, 0.9723724797711119, 0.9714336805400331]
validation_cnn = [0.67, 0.79, 0.795, 0.795, 0.83, 0.805, 0.795, 0.785, 0.795, 0.805]

epochs = range(1, len(training) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, training, label='Training Accuracy', marker='o')
plt.plot(epochs, validation, label='Validation Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

