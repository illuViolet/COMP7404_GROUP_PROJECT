import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import resample

# prepare datasets
def load_data(data_root):
    paths, labels = [], []
    classes = sorted(os.listdir(os.path.join(data_root, "images")))
    for cls in classes:
        cls_dir = os.path.join(data_root, "images", cls)
        # choose the first 20 figures
        for img_name in os.listdir(cls_dir)[:30]:
            img_path = os.path.join(cls_dir, img_name)
            paths.append(img_path)
            labels.append(cls)
    return paths, LabelEncoder().fit_transform(labels)

# extract features using RESNET50
def init_feature_extractor():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

def extract_features(paths, model):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    features = []
    for path in paths:
        img = Image.open(path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(img_tensor).flatten().numpy()
        features.append(feat)
    return np.array(features)

# load datas
paths, labels = load_data("dataset/amazon")

# extract features
model = init_feature_extractor()
features = extract_features(paths, model)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# PCA
extended_data = np.vstack([scaled_features, scaled_features])
pca = PCA(n_components=800)
X = pca.fit_transform(extended_data)
n = len(X)
X = X[:n//2, :]
labels = labels+1

# save data
np.save('Task1_datas/amazon_features', X)
np.save('Task1_datas/amazon_labels', labels)

# # cut target set and test set
# # get all the categories
# unique_classes = np.unique(labels)
# X_filtered = []
# y_filtered = []
# X_remaining = []
# y_remaining = []
#
# for label in unique_classes:
#     X_class = X[labels == label]
#     y_class = labels[labels == label]
#
#     if X_class.shape[0] > 3:
#         X_class_resampled, y_class_resampled = resample(X_class, y_class,
#                                                        n_samples=3,
#                                                        random_state=42)
#         _, selected_indices = resample(np.arange(len(X_class)),
#                                        np.arange(len(X_class)),
#                                        n_samples=3,
#                                        random_state=42,
#                                        stratify=None)
#         mask_selected = np.zeros(len(X_class), dtype=bool)
#         mask_selected[selected_indices] = True
#         X_remaining.append(X_class[~mask_selected])
#         y_remaining.append(y_class[~mask_selected])
#     else:
#         X_class_resampled, y_class_resampled = X_class, y_class
#
#     X_filtered.append(X_class_resampled)
#     y_filtered.append(y_class_resampled)
#
# X_filtered = np.vstack(X_filtered)
# y_filtered = np.concatenate(y_filtered)
# X_remaining = np.vstack(X_remaining)
# y_remaining = np.concatenate(y_remaining)
#
# print("Shape of filtered X_filtered:", X_filtered.shape)
# print("Shape of filtered y_filtered:", y_filtered.shape)
#
# # save X and y
# np.save('Task1_datas/target_set_features_dslr.npy', X_filtered)
# np.save('Task1_datas/target_set_labels_dslr.npy', y_filtered)
#
# print("Shape of filtered X_remaining:", X_remaining.shape)
# print("Shape of filtered y_remaining:", y_remaining.shape)
#
# # save X and y
# np.save('Task1_datas/test_set_features_dslr.npy', X_remaining)
# np.save('Task1_datas/test_set_labels_dslr.npy', y_remaining)


# # SVM test
# X_train, X_test, y_train, y_test = train_test_split(
#     scaled_features, labels, test_size=0.2, random_state=42
# )
# svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# print(f"Base Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01, 0.1]}
# grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
# grid.fit(X_train, y_train)
# best_svm = grid.best_estimator_
# print(f"Best Params: {grid.best_params_}\nBest Accuracy: {grid.best_score_:.4f}")
#
# ConfusionMatrixDisplay.from_estimator(best_svm, X_test, y_test,
#                                      cmap='Blues',
#                                      xticks_rotation=90)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()
