function [kernelWeights, svmModel, finalObjective, combinedKernel] = LPMKL_labels(labels, kernelMatrix, kernelRoot, transformationVectors, parameters)

[nSamples, nComponents] = size(transformationVectors);
assert(nSamples == size(labels, 1), 'Labels and transformation vectors must have matching dimensions');
assert(nComponents == size(labels, 2), 'Number of label components must match transformation vectors');

% Set default parameters
defaultDegree = 1;
defaultNormConstraint = 1;
defaultWeights = ones(nSamples, 1);

% Override defaults with provided parameters
if isfield(parameters, 'degree')
    defaultDegree = parameters.degree;
end
if isfield(parameters, 'd_norm')
    defaultNormConstraint = parameters.d_norm;
end

%% Optimization Parameters
maxIterations = 100;      % Maximum iterations for MKL optimization
convergenceThreshold = 1e-3; % Relative change threshold for convergence

%% Initialize Kernel Weights
if isfield(parameters, 'd')
    % Normalize provided initial weights
    initialWeights = parameters.d;
    normalizedWeights = defaultNormConstraint * initialWeights / (sum(initialWeights.^defaultDegree)^(1/defaultDegree));
    kernelWeights = zeros(nComponents, 1);
    kernelWeights(1:length(normalizedWeights)) = normalizedWeights;
else
    % Uniform initialization if no weights provided
    kernelWeights = defaultNormConstraint * ones(nComponents, 1) * (1/nComponents)^(1/defaultDegree);
end

%% Main Optimization Loop
objectiveHistory = [];

% First iteration
[svmModel, objectiveHistory(1), weightNorm, combinedKernel] = ...
    solveSVMWithKernel(kernelMatrix, kernelRoot, transformationVectors, labels, kernelWeights, parameters);

for iter = 2:maxIterations
    % Update kernel weights using LpMKL formulation
    exponent = 2/(defaultDegree + 1);
    weightedNorm = weightNorm.^exponent;
    normalizationFactor = (sum(weightedNorm.^defaultDegree))^(1/defaultDegree);
    kernelWeights = defaultNormConstraint * weightedNorm / normalizationFactor;
    
    % Solve SVM with current kernel weights
    [svmModel, objectiveHistory(iter), weightNorm, combinedKernel] = ...
        solveSVMWithKernel(kernelMatrix, kernelRoot, transformationVectors, labels, kernelWeights, parameters);
    
    % Check convergence
    if abs(objectiveHistory(iter) - objectiveHistory(iter-1)) <= convergenceThreshold * abs(objectiveHistory(iter))
        break;
    end
end

finalObjective = objectiveHistory(end);
end


function [model, objective, weightNorm, kernel] = solveSVMWithKernel(K, K_root, H, labels, weights, params)
% Solves the SVM subproblem with given kernel weights and label information

[nSamples, nComponents] = size(H);

% Combine kernels using current weights
kernel = combineKernelsWithLabels(K, K_root, H, labels, weights) + diag(params.weight);

% Train SVM with combined kernel
svmOptions = ['-q -s 2 -t 4 -n ', num2str(1/nSamples)];
model = svmtrain(ones(nSamples, 1), ones(nSamples, 1), [(1:nSamples)', kernel], svmOptions);

% Extract support vector information
supportVectorIndices = full(model.SVs);
dualCoefficients = abs(model.sv_coef);

% Compute kernel projections
supportVectorsKernel = K_root(supportVectorIndices, :) * H;
weightedLabels = repmat(dualCoefficients, [1 nComponents]) .* labels(supportVectorIndices, :);

% Compute quadratic terms for objective function
quadraticTerms = (sum(supportVectorsKernel .* weightedLabels).^2) + sum((K_root(:, supportVectorIndices) * weightedLabels).^2) + (sum(weightedLabels).^2);
quadraticTerms = quadraticTerms';

% Compute objective value
objective = -0.5 * (sum(quadraticTerms .* weights) + dualCoefficients' * (dualCoefficients .* params.weight(supportVectorIndices)));

% Compute weight normalization terms
weightNorm = weights .* sqrt(quadraticTerms);
end


function kernel = combineKernelsWithLabels(K, K_root, H, labels, weights)
% Combines multiple kernels using weights and label information

[nSamples, nComponents] = size(H);
kernel = zeros(nSamples);
for i = 1:nComponents
    kernel = kernel + weights(i) * (K_root * (H(:,i) * H(:,i)') * K_root) .* (labels(:,i) * labels(:,i)');
end
kernel = kernel + K;  % Add the base kernel
end