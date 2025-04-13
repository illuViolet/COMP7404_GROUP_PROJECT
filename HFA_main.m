clear; clc;

%% Add Required Paths
libsvmWeightedPath = '.\libs\libsvm-weights-3.20\matlab';
commonFunctionsPath = '.\commfuns';

addpath(libsvmWeightedPath);
addpath(commonFunctionsPath);

%% Data Loading and Initial Processing
% Define document categories for classification
load('Task1_datas/categories.mat')

% Initialize Python numpy interface for loading .npy files
np = py.importlib.import_module('numpy');

% Load feature matrices and labels
sourceDomainFeatures = double(np.load('Task1_datas/amazon_features.npy'))';
sourceDomainLabels = double(np.load('Task1_datas/amazon_labels.npy'))';
targetDomainFeatures = double(np.load('Task1_datas/target_set_features_dslr.npy'))';
targetDomainLabels = double(np.load('Task1_datas/target_set_labels_dslr.npy'))';
testSetFeatures = double(np.load('Task1_datas/test_set_features_dslr.npy'))';
testSetLabels = double(np.load('Task1_datas/test_set_labels_dslr.npy'))';

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

%% Kernel Matrix Computation
% Configure Gaussian kernel parameters
kernelParams.kernel_type = 'gaussian';

% Compute kernel matrices
[sourceKernelMatrix, sourceKernelParams] = getKernel(sourceDomainFeatures, kernelParams);
[targetKernelMatrix, targetKernelParams] = getKernel(targetDomainFeatures, kernelParams);

% Compute matrix square roots for kernel matrices
[sourceKernelRoot, ~] = sqrtm(sourceKernelMatrix); 
sourceKernelRoot = real(sourceKernelRoot);
[targetKernelRoot, ~] = sqrtm(targetKernelMatrix); 
targetKernelRoot = real(targetKernelRoot);

% Get sample counts
numSourceSamples = size(sourceKernelMatrix, 1);
numTargetSamples = size(targetKernelMatrix, 1);
totalSamples = numSourceSamples + numTargetSamples;

% Construct combined kernel matrix
combinedKernelRoot = [sourceKernelRoot zeros(numSourceSamples, numTargetSamples); 
                     zeros(numTargetSamples, numSourceSamples) targetKernelRoot];

% Compute pseudoinverse for target kernel
targetKernelRootInv = real(pinv(targetKernelRoot));

% Construct transformation operator
targetProjectionOperator = [zeros(numSourceSamples, numTargetSamples); 
                           eye(numTargetSamples)] * targetKernelRootInv;

% Compute test kernel matrix
testKernelMatrix = getKernel(testSetFeatures, targetDomainFeatures, targetKernelParams);

%% Domain Adaptation Classification
% Set algorithm hyperparameters
hfaParams.sourceRegularization = 1;      
hfaParams.targetRegularization = 1;      
hfaParams.lambda = 100;                  
hfaParams.mklDegree = 1;              

% Initialize decision values storage
classDecisionValues = zeros(length(testSetLabels), length(documentCategories)-1);

for categoryIdx = 1:length(documentCategories)
    fprintf('Processing category %d: %s\n', categoryIdx, documentCategories{categoryIdx});
    
    % Create binary labels for current category
    sourceBinaryLabels = 2*(sourceDomainLabels == categoryIdx) - 1;
    targetBinaryLabels = 2*(targetDomainLabels == categoryIdx) - 1;
    
    %% Model Training
    [trainedModel, transformationMatrix, objectiveHistory] = train_hfa_modified(sourceBinaryLabels, targetBinaryLabels, combinedKernelRoot, hfaParams);
    
    %% Prediction Computation
    decisionThreshold = trainedModel.rho * trainedModel.Label(1);
    
    % Extract and format support vector coefficients
    dualCoefficients = zeros(totalSamples, 1);
    dualCoefficients(full(trainedModel.SVs)) = trainedModel.sv_coef;
    dualCoefficients = dualCoefficients * trainedModel.Label(1);
    
    % Get target domain specific coefficients
    targetDomainCoefficients = dualCoefficients(numSourceSamples+1:end);
    
    % Compute final decision values
    kernelProjection = testKernelMatrix * targetProjectionOperator';
    transformedFeatures = kernelProjection * transformationMatrix * combinedKernelRoot;
    classDecisionValues(:, categoryIdx) = transformedFeatures * dualCoefficients + ...
                                         testKernelMatrix * targetDomainCoefficients - ...
                                         decisionThreshold;
end

%% Performance Evaluation
% Determine predicted classes
[~, predictedLabels] = max(classDecisionValues, [], 2);

% Calculate accuracy
classificationAccuracy = sum(predictedLabels == testSetLabels)/length(testSetLabels);
fprintf('Final classification accuracy: %.4f\n', classificationAccuracy);

% =========================================================================
%% Confusion Matrix Visualization
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