function [trainedModel, transformationMatrix, objectiveValues] = train_hfa_modified(sourceLabels, targetLabels, kernelMatrixRoot, parameters)

MAX_ITERATIONS = 50;       % Maximum number of optimization iterations
CONVERGENCE_THRESHOLD = 1e-3; % Relative change threshold for convergence

% Override defaults if specified in parameters
if isfield(parameters, 'hfa_iter')
    MAX_ITERATIONS = parameters.hfa_iter;
end

if isfield(parameters, 'hfa_tau')
    CONVERGENCE_THRESHOLD = parameters.hfa_tau;
end

%% Data Dimension Validation
nSourceSamples = length(sourceLabels);
nTargetSamples = length(targetLabels);
totalSamples = size(kernelMatrixRoot, 1);

% Verify kernel matrix dimensions match label counts
assert(totalSamples == nSourceSamples + nTargetSamples, 'Kernel matrix dimensions do not match label counts');

%% Instance Weighting Setup
% Create weight vector with different regularization for source/target
instanceWeights = [ones(nSourceSamples, 1) * parameters.sourceRegularization; 
                   ones(nTargetSamples, 1) * parameters.targetRegularization];
combinedLabels = [sourceLabels; targetLabels];

%% HFA-MKL Training Initialization
objectiveValues = [];  % Stores objective function values during optimization

% Initialize transformation vectors with uniform values scaled by lambda
transformationVectors = sqrt(parameters.lambda) * ones(totalSamples, 1) / sqrt(totalSamples);

% Configure MKL parameters
mklParameters.svm.C = 1;                      % Base SVM regularization
mklParameters.d_norm = 1;                     % Norm constraint on kernel weights
mklParameters.degree = parameters.mklDegree; % p-norm degree for MKL
mklParameters.weight = instanceWeights;       % Instance weights for weighted SVM

%% Main Optimization Loop
converged = false;
for iteration = 1:MAX_ITERATIONS
    % Print current iteration
    fprintf('\tIteration #%-2d:\n', iteration);
    
    %% Solve MKL with current transformation
    [kernelWeights, currentModel, currentObjective] = LPMKL_fast(...
        combinedLabels, ...
        kernelMatrixRoot, ...
        transformationVectors, ...
        mklParameters);
    
    % Store current results
    objectiveValues(iteration) = currentObjective;
    trainedModel = currentModel;
    
    %% Display optimization progress
    if iteration == 1
        fprintf('Initial objective = %.15f\n', currentObjective);
    else
        objectiveChange = abs(currentObjective - objectiveValues(iteration-1));
        fprintf('Current objective = %.15f, Change = %.15f\n', currentObjective, objectiveChange);
    end
    
    %% Prepare for transformation update
    % Get SVM dual coefficients (alpha)
    dualCoefficients = zeros(totalSamples, 1);
    supportVectorIndices = full(trainedModel.SVs);
    dualCoefficients(supportVectorIndices) = abs(trainedModel.sv_coef);
    
    %% Check convergence
    if iteration > 1
        relativeChange = abs(currentObjective - objectiveValues(iteration-1)) / abs(currentObjective);
        
        converged = (relativeChange <= CONVERGENCE_THRESHOLD) || (iteration == MAX_ITERATIONS);
    end
    
    if converged
        break;
    end
    
    %% Update transformation vectors
    % Compute weighted kernel projection
    weightedCoefficients = combinedLabels .* dualCoefficients;
    kernelProjection = kernelMatrixRoot * weightedCoefficients;
    
    % Normalize and scale new direction
    projectionNorm = sqrt(kernelProjection' * kernelProjection);
    newDirection = sqrt(parameters.lambda) * (kernelProjection / projectionNorm);
    
    % Append new direction to transformation vectors
    transformationVectors = [transformationVectors, newDirection];
end

%% Transformation Matrix Computation
% Combine transformation vectors with learned kernel weights
weightedTransformationVectors = transformationVectors;
for dim = 1:length(kernelWeights)
    weightedTransformationVectors(:, dim) = transformationVectors(:, dim) * sqrt(kernelWeights(dim));
end

% Compute final transformation matrix
transformationMatrix = weightedTransformationVectors * weightedTransformationVectors';
end