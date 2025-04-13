function [kernelWeights, svmModel, finalObjective] = LPMKL_fast(labels, kernelRoot, transformationVectors, parameters)

[nSamples, nComponents] = size(transformationVectors);

%% Initialize Parameters
% Set default values if not provided
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
if isfield(parameters, 'weight')
    defaultWeights = parameters.weight;
end

% Store parameters
algorithmParams = parameters;
algorithmParams.weight = defaultWeights;

%% Initialize Kernel Weights
% Convergence criteria
maxIterations = 100;
convergenceThreshold = 1e-3;

% Initialize kernel weights
if isfield(parameters, 'd')
    % Use provided initial weights with normalization
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
[svmModel, objectiveHistory(1), weightNorm] = solveSVMSubproblem(kernelRoot, labels, transformationVectors, kernelWeights, algorithmParams);

for iter = 2:maxIterations
    % Update kernel weights
    exponent = 2/(defaultDegree + 1);
    weightedNorm = weightNorm.^exponent;
    normalizationFactor = (sum(weightedNorm.^defaultDegree))^(1/defaultDegree);
    kernelWeights = defaultNormConstraint * weightedNorm / normalizationFactor;
    
    % Solve SVM with current kernel weights
    [svmModel, objectiveHistory(iter), weightNorm] = solveSVMSubproblem(kernelRoot, labels, transformationVectors, kernelWeights, algorithmParams);
    
    % Check convergence
    if abs(objectiveHistory(iter) - objectiveHistory(iter-1)) <= convergenceThreshold * abs(objectiveHistory(iter))
        break;
    end
end

finalObjective = objectiveHistory(end);
end


function [model, objective, weightNorm] = solveSVMSubproblem(kernelRoot, labels, basisVectors, weights, params)
% Solves the SVM subproblem for given kernel weights

[nSamples, nComponents] = size(basisVectors);

% Construct combined kernel matrix
combinedKernel = combineKernels(kernelRoot, basisVectors, weights);

% Train weighted SVM
svmOptions = ['-t 4 -q -c ', num2str(params.svm.C)];
model = svmtrain(params.weight, labels, [(1:size(combinedKernel, 1))', combinedKernel], svmOptions);

% Extract dual coefficients
supportVectorIndices = full(model.SVs);
dualCoefficients = zeros(nSamples, 1);
dualCoefficients(supportVectorIndices) = abs(model.sv_coef);

% Compute objective function components
kernelProjection = kernelRoot * (dualCoefficients .* labels);
basisProjection = basisVectors' * kernelProjection;
squaredNorms = basisProjection.^2;

% Compute objective value
objective = sum(dualCoefficients) - 0.5 * (sum(squaredNorms .* weights) + kernelProjection' * kernelProjection);

% Compute weight normalization terms
weightNorm = weights .* sqrt(squaredNorms);
end


function kernelMatrix = combineKernels(kernelRoot, basisVectors, weights)
% Combines multiple kernels using learned weights

[nSamples, nComponents] = size(basisVectors);

% Weight the basis vectors
weightedBasis = basisVectors;
for i = 1:nComponents
    weightedBasis(:, i) = basisVectors(:, i) * sqrt(weights(i));
end

% Construct transformation matrix
transformationMatrix = weightedBasis * weightedBasis';

% Final kernel construction
kernelMatrix = kernelRoot * (transformationMatrix + eye(nSamples)) * kernelRoot;
end