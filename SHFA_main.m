clear; clc;
% =============================================
% set paths
addpath('.\libs\libsvm-weights-3.20\matlab');
addpath('.\libs\sumkernels');
addpath('.\commfuns');

% =============================================
% set params
param.C_s = 1;
param.C_t = 1;
param.C_x = 1e-3;
param.sigma         = 100;
param.mkl_degree    = 1.5;
param.ratio_var     = 0;
param.hfa_iter      = 50;
param.hfa_tau       = 0.001;

% =============================================
load('Task2_datas/categories.mat')
categories = documentCategories;

np = py.importlib.import_module('numpy');

% Load feature matrices and labels
sourceDomainFeatures = double(np.load('Task2_datas/X_TF_EN-GR.npy'))';
sourceDomainLabels = double(np.load('Task2_datas/y_labels_EN-GR.npy'))';
targetDomainFeatures = double(np.load('Task2_datas/X_TF_EN-SP.npy'))';
targetDomainLabels = double(np.load('Task2_datas/y_labels_EN-SP.npy'))';
testSetFeatures = double(np.load('Task2_datas/X_TF_EN-SP_test_samples.npy'))';
testSetLabels = double(np.load('Task2_datas/y_labels_EN-SP_test_samples.npy'))';


% Shuffle source data with fixed seed for reproducibility
rng(42);  
sourceIndices = randperm(length(sourceDomainLabels));  
sourceDomainFeatures = sourceDomainFeatures(:, sourceIndices);
sourceDomainLabels = sourceDomainLabels(sourceIndices);

% Shuffle target data
rng(42);  
targetIndices = randperm(length(targetDomainLabels));  
targetDomainFeatures = targetDomainFeatures(:, targetIndices);
targetDomainLabels = targetDomainLabels(targetIndices);

% Shuffle test data
rng(42);  
testIndices = randperm(length(testSetLabels));  
testSetFeatures = testSetFeatures(:, testIndices);
testSetLabels = testSetLabels(testIndices);

target_features = [targetDomainFeatures testSetFeatures];
% =============================================
% prepare kernels
kparam.kernel_type =  'gaussian';
[K_s, param_s] = getKernel(sourceDomainFeatures, kparam);
[K_t, param_t] = getKernel(target_features, kparam);

[K_s_root, resnorm_s] = sqrtm(K_s); K_s_root = real(K_s_root);
[K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
n_s = size(K_s, 1);
n_t = size(K_t, 1);

K       = [K_s zeros(n_s, n_t); zeros(n_t, n_s) K_t];
K_root  = [K_s_root zeros(n_s, n_t); zeros(n_t, n_s) K_t_root];

K_t_root_inv = real(pinv(K_t_root));
L_t_inv = [zeros(n_s, n_t); eye(n_t)] * K_t_root_inv;

% do kernel decomposition for inference \y
aug_features    = sqrtm((1+param.sigma)*K+ones(size(K)));
aug_features    = real(aug_features);

% =========================================================================
% train one-versus-all classifiers
for c = 1:length(categories)
    fprintf(1, '-- Class %d: %s\n', c, categories{c});
    source_labels       = 2*(sourceDomainLabels == c) - 1;
    target_labels       = 2*(targetDomainLabels == c) - 1;
    
    ratio               = sum(testSetLabels == c)/length(testSetLabels);
    param.upper_ratio   = ratio;
    param.lower_ratio   = ratio;    
    
    % training
    [model, Us, labels, coefficients, rho, obj] = train_shfa_modified(source_labels, target_labels, K, K_root, aug_features, param);
    % testing
    K_test                  = getKernel(testSetFeatures, target_features, param_t);
    dec_values(:, c)        = predict_ifa_semi_kernel(K_test, model, Us, labels, coefficients, rho, K_root, L_t_inv);    
end

% =========================================================================
% display results
test_labels         = testSetLabels;
[~, predict_labels] = max(dec_values, [], 2);
acc     =  sum(predict_labels == test_labels)/length(test_labels);
fprintf('SHFA accuracy = %f\n', acc);

%% Confusion Matrix Visualization
predictedLabels = predict_labels;
confusionMatrix = confusionmat(testSetLabels, predictedLabels);

myColormap = [linspace(0.8, 1, 256)' linspace(0.9, 1, 256)' ones(256, 1)];
myColormap = flipud(myColormap); 
figure;
heatmap(documentCategories, documentCategories, confusionMatrix, ...
    'Colormap', myColormap, ...
    'ColorbarVisible', 'on', ...
    'CellLabelColor', 'k');

title('Confusion Matrix');
xlabel('Predicted Categories');
ylabel('Actual Categories');

% calculate accuracy, recall and F1 score
nClasses = max(testSetLabels); 
accuracy_per_class = zeros(nClasses, 1);
recall_per_class = zeros(nClasses, 1);
f1_per_class = zeros(nClasses, 1);

macro_precision = 0;
macro_recall = 0;
macro_f1 = 0;

for classIdx = 1:nClasses
    truePositives = sum((testSetLabels == classIdx) & (predictedLabels == classIdx));
    falsePositives = sum((testSetLabels ~= classIdx) & (predictedLabels == classIdx));
    falseNegatives = sum((testSetLabels == classIdx) & (predictedLabels ~= classIdx));
    trueNegatives = sum((testSetLabels ~= classIdx) & (predictedLabels ~= classIdx));
    accuracy_per_class(classIdx) = truePositives / sum(testSetLabels == classIdx);
    recall_per_class(classIdx) = truePositives / (truePositives + falseNegatives);
    precision = truePositives / (truePositives + falsePositives);
    f1_per_class(classIdx) = 2 * (precision * recall_per_class(classIdx)) / (precision + recall_per_class(classIdx));
    macro_precision = macro_precision + precision;
    macro_recall = macro_recall + recall_per_class(classIdx);
    macro_f1 = macro_f1 + f1_per_class(classIdx);
end
macro_accuracy = mean(accuracy_per_class);
macro_recall = macro_recall / nClasses;
macro_f1 = macro_f1 / nClasses;

fprintf('Accuracy per class:\n');
disp(accuracy_per_class);
fprintf('Recall per class:\n');
disp(recall_per_class);
fprintf('F1 Score per class:\n');
disp(f1_per_class);
fprintf('Macro-average Accuracy: %.4f\n', macro_accuracy);
fprintf('Macro-average Recall: %.4f\n', macro_recall);
fprintf('Macro-average F1 Score: %.4f\n', macro_f1);
