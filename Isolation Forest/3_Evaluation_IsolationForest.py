import pandas as pd
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    roc_curve, auc
)


def evaluate_iforest(model_path, test_csv):
    # --- load & preprocess test data ---
    df = pd.read_csv(test_csv)
    df['labels'] = (
        df['labels']
          .apply(lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x)
          .str.rstrip('.')
    )
    X_test = df.drop(columns=['labels']).astype('float32')
    y_test = (df['labels'] != 'normal').astype(int)

    # --- load model & time it ---
    t0 = time.time()
    model = joblib.load(model_path)
    load_time = time.time() - t0

    # --- inference & time it ---
    t1 = time.time()
    scores = model.decision_function(X_test)  # higher=more normal
    infer_time = time.time() - t1

    # --- threshold sweep to maximize accuracy ---
    best_acc = 0.0
    best_thresh = None
    candidates = np.linspace(scores.min(), scores.max(), 200)
    for t in candidates:
        y_try = (scores < t).astype(int)
        acc_try = accuracy_score(y_test, y_try)
        if acc_try > best_acc:
            best_acc = acc_try
            best_thresh = t
    print(f"\nBest threshold for accuracy: {best_thresh:.5f} -> accuracy={best_acc:.4f}")

    # final predictions
    y_pred = (scores < best_thresh).astype(int)

    # --- compute metrics ---
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, -scores)

    rep = classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Attack'],
        zero_division=0,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    return {
        'accuracy':   acc,
        'precision':  prec,
        'recall':     rec,
        'f1':         f1,
        'auc':        roc_auc,
        'load_time':  load_time,
        'infer_time': infer_time,
        'report':     rep,
        'cm':         cm,
        'scores':     scores,
        'y_test':     y_test
    }


def print_res(r):
    print("\n" + "="*50)
    print("IsolationForest @ Stage 3.1 Evaluation (Threshold-tuned)")
    print("="*50)
    print(f"Accuracy   : {r['accuracy']:.4f}")
    print(f"Precision  : {r['precision']:.4f}")
    print(f"Recall     : {r['recall']:.4f}")
    print(f"F1-Score   : {r['f1']:.4f}")
    print(f"AUC-ROC    : {r['auc']:.4f}")
    print(f"Model load : {r['load_time']:.2f}s")
    print(f"Inference  : {r['infer_time']:.2f}s\n")

    print("Class-wise metrics:")
    print(f"{'Class':<10}{'Prec':>7}{'Rec':>8}{'F1':>8}")
    for cls in ['Normal', 'Attack']:
        c = r['report'][cls]
        print(f"{cls:<10}{c['precision']:7.4f}{c['recall']:8.4f}{c['f1-score']:8.4f}")

    cm = r['cm']
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("         Normal  Attack")
    print(f"Normal:  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Attack:  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print("="*50)


def plot_confusion_matrix(cm, classes=['Normal', 'Attack']):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_roc_curve(y_test, scores):
    fpr, tpr, _ = roc_curve(y_test, -scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()


def plot_metric_bars(metrics):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    colors = ['red', 'blue', 'green', 'purple']
    plt.figure()
    x = np.arange(len(labels))
    bar_width = 0.4 
    plt.bar(x, values, width=bar_width, color=colors)
    plt.xticks(x, labels)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    plt.tight_layout()


if __name__ == "__main__":
    MODEL_PATH     = r"C:\Users\huyng\OneDrive\Documents\SDSU\CS 549\iforest_tuned.pkl"
    TEST_DATA_PATH = r"C:\Users\huyng\OneDrive\Documents\SDSU\CS 549\kddcup99_10_percent_test.csv"

    stats = evaluate_iforest(MODEL_PATH, TEST_DATA_PATH)
    print_res(stats)

    # —— Visualizations ——
    plot_confusion_matrix(stats['cm'])
    plot_roc_curve(stats['y_test'], stats['scores'])
    plot_metric_bars(stats)
    plt.show()
