import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss
from sklearn.decomposition import PCA
from scipy.sparse import save_npz, load_npz
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer  # 新增导入
import matplotlib
matplotlib.use('TkAgg')

categories = {'C15': 4587, 'CCAT': 8745, 'E21': 9625, 'ECAT': 5656, 'GCAT': 5745, 'M11': 45845}

def filter_datas(file, target_file):
    with open(file, 'r') as document_read:
        with open(target_file, 'w') as document_write:
            for line in document_read:
                target = line.split(None, 1)[0]
                # 保留原始文本内容（原特征部分保持不变）
                line_to_write = f'{categories[target]} {line[len(target):].strip()}\n'
                document_write.write(line_to_write)
    return target_file

file = filter_datas('rcv1rcv2aminigoutte/SP/Index_EN-SP', 'svml_en_fr.txt')

y = []
corpus = []
with open(file, 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            label, text = parts
            y.append(int(label))
            corpus.append(text)

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=1,
    max_df=0.8
)
X = vectorizer.fit_transform(corpus)

print('Data matrix shape after TF-IDF:', X.shape)

# PCA
pca = PCA(n_components=807) # EN 1131; FR 1230; GR 1417; IT 1041; SP 807
X = pca.fit_transform(X.toarray())

y = np.array(y)
label_mapping = {
    4587: 1,
    8745: 2,
    9625: 3,
    5656: 4,
    5745: 5,
    45845: 6
}

y = np.array([label_mapping[i] for i in y])

# cut features
X_filtered = []
y_filtered = []
# the remaining sets
X_remaining = []
y_remaining = []
# get all the categories
unique_classes = np.unique(y)

for label in unique_classes:
    X_class = X[y == label]
    y_class = y[y == label]

    # if sample numbers > 100
    if X_class.shape[0] > 100:
        X_class_resampled, y_class_resampled = resample(X_class, y_class,
                                                       n_samples=100,
                                                       random_state=42)
        _, selected_indices = resample(np.arange(len(X_class)),
                                       np.arange(len(X_class)),
                                       n_samples=3,
                                       random_state=42,
                                       stratify=None)
        mask_selected = np.zeros(len(X_class), dtype=bool)
        mask_selected[selected_indices] = True
        X_remaining.append(X_class[~mask_selected])
        y_remaining.append(y_class[~mask_selected])
    else:
        X_class_resampled, y_class_resampled = X_class, y_class
    X_filtered.append(X_class_resampled)
    y_filtered.append(y_class_resampled)

X_filtered = np.vstack(X_filtered)
y_filtered = np.concatenate(y_filtered)
X_remaining = np.vstack(X_remaining)
y_remaining = np.concatenate(y_remaining)

print("Shape of filtered X:", X_filtered.shape)
print("Shape of filtered y:", y_filtered.shape)

# save X and y
np.save('Task2_datas/X_tf_EN-SP.npy', X_filtered)
np.save('Task2_datas/y_labels_EN-SP.npy', y_filtered)

# the remaining sets (perclass is 40)
n_samples_per_class = 40

X_resampled_list = []
y_resampled_list = []

for label in unique_classes:
    X_class = X_remaining[y_remaining == label]
    y_class = y_remaining[y_remaining == label]
    X_class_resampled, y_class_resampled = resample(X_class, y_class,
                                                    n_samples=n_samples_per_class,
                                                    random_state=42)
    X_resampled_list.append(X_class_resampled)
    y_resampled_list.append(y_class_resampled)

X_resampled = np.vstack(X_resampled_list)
y_resampled = np.concatenate(y_resampled_list)

np.save('Task2_datas/X_tf_EN-SP_test_samples.npy', X_resampled)
np.save('Task2_datas/y_labels_EN-SP_test_samples.npy', y_resampled)

print("Shape of X_resampled:", X_resampled.shape)
print("Shape of y_resampled:", y_resampled.shape)

# # SVM test
# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.25,
#     random_state=42,
#     stratify=y
# )
# def plot_confusion_matrix(cm, classes,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j]),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
# svc = SVC()
# svc.fit(X_train, y_train)
# y_pred_SVC = svc.predict(X_test)
# class_as_number = [4587, 8745, 9625, 5656, 5745, 45845]
# confusion_SVC = confusion_matrix(y_test, y_pred_SVC, labels=class_as_number)
# class_as_string = ['C15', 'CCAT', 'E21', 'ECAT', 'GCAT', 'M11']
# plt.figure()
# plot_confusion_matrix(confusion_SVC, classes=class_as_string)
# plt.show()
# report_SVC = classification_report(y_test, y_pred_SVC, labels=class_as_number, target_names=class_as_string)
# print(report_SVC)
