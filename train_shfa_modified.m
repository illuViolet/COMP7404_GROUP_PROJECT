function [model, transformationVectors, labelMatrix, kernelWeights, decisionThreshold, objectiveHistory] = train_shfa_modified(sourceLabels, targetLabels, kernelMatrix, kernelRoot, featureEmbeddings, parameters)

% Set default parameters
maxIterations = 20;
convergenceThreshold = 1e-3;

% Override defaults if specified
if isfield(parameters, 'hfa_iter')
    maxIterations = parameters.hfa_iter;
end
if isfield(parameters, 'hfa_tau')
    convergenceThreshold = parameters.hfa_tau;
end

%% Data Preparation
nSource = length(sourceLabels);
nLabeledTarget = length(targetLabels);
nTotal = size(kernelMatrix, 1);
nUnlabeled = nTotal - nSource - nLabeledTarget;

% Create instance weights based on domain
instanceWeights = [
    ones(nSource, 1) * (1/parameters.C_s);
    ones(nLabeledTarget, 1) * (1/parameters.C_t);
    ones(nUnlabeled, 1) * (1/parameters.C_x)
];

% Train initial SVM on labeled target data
targetKernel = (kernelMatrix(nSource+1:nSource+nLabeledTarget, ...
                nSource+1:nSource+nLabeledTarget) + 1) .* ...
               (targetLabels * targetLabels') + ...
               diag(instanceWeights(nSource+1:nSource+nLabeledTarget));

svmOptions = ['-q -s 2 -t 4 -n ', num2str(1/nLabeledTarget)];
initialModel = svmtrain(ones(nLabeledTarget,1), ones(nLabeledTarget,1), [(1:nLabeledTarget)', targetKernel], svmOptions);

% Extract support vector coefficients
dualCoefficients = zeros(nLabeledTarget, 1);
dualCoefficients(full(initialModel.SVs)) = abs(initialModel.sv_coef);
weightedLabels = targetLabels .* dualCoefficients;

% Predict labels for unlabeled data
unlabeledScores = (kernelMatrix(nSource+nLabeledTarget+1:end, ...
                   nSource+1:nSource+nLabeledTarget)+1) * weightedLabels;
estimatedLabels = (unlabeledScores > 0);

% Apply ratio constraints to unlabeled predictions
[sortedScores, sortIndices] = sort(unlabeledScores, 'descend');
if sum(estimatedLabels) > parameters.upper_ratio * nUnlabeled
    estimatedLabels(sortIndices(floor(parameters.upper_ratio*nUnlabeled)+1:end)) = 0;
elseif sum(estimatedLabels) < parameters.lower_ratio * nUnlabeled
    estimatedLabels(sortIndices(1:ceil(parameters.lower_ratio*nUnlabeled))) = 1;
end

% Convert to {-1, +1} format
estimatedLabels = 2*estimatedLabels - 1;
labelMatrix = [sourceLabels; targetLabels; estimatedLabels];

%% Main SHFA Training Loop
objectiveHistory = [];
transformationVectors = sqrt(parameters.sigma) * ones(nTotal, 1) / sqrt(nTotal);

% Configure MKL parameters
mklParams.d_norm = 1;
mklParams.degree = parameters.mkl_degree;
mklParams.weight = instanceWeights;

for iter = 1:maxIterations
    fprintf('\tIteration #%-2d:\n', iter);
    
    % Solve MKL problem with current transformation
    [kernelWeights, model, currentObjective, combinedKernel] = LPMKL_labels(labelMatrix, kernelMatrix, kernelRoot, transformationVectors, mklParams);

    % Store optimization progress
    objectiveHistory(iter) = currentObjective;
    
    % Display progress
    if iter > 1
        fprintf('Objective = %.15f, Change = %.15f\n', currentObjective, abs(currentObjective - objectiveHistory(iter-1)));
    else
        fprintf('Objective = %.15f\n', currentObjective);
    end

    % Extract support vector coefficients
    dualCoefficients = zeros(nTotal, 1);
    dualCoefficients(full(model.SVs)) = abs(model.sv_coef);

    % Check convergence
    if (iter > 1 && abs(currentObjective - objectiveHistory(iter-1)) <= convergenceThreshold * abs(currentObjective)) || (iter == maxIterations)
        break;
    end
    
    %% Update Transformation Vectors
    % Process feature embeddings with current coefficients
    embeddingDimension = size(featureEmbeddings, 1);
    weightedEmbeddings = featureEmbeddings' .* repmat(dualCoefficients, [1, embeddingDimension]);
    unlabeledWeights = weightedEmbeddings(nSource+nLabeledTarget+1:end, :);
    
    % Find most violated constraints
    [~, posIndices] = sort(unlabeledWeights, 'descend');
    [~, negIndices] = sort(-unlabeledWeights, 'descend');
    sortIndices = [posIndices negIndices];
    
    candidateLabels = [unlabeledWeights, -unlabeledWeights] > 0;
    
    % Apply ratio constraints to candidate labels
    for dim = 1:2*embeddingDimension
        currentLabels = candidateLabels(:, dim);
        if sum(currentLabels) > parameters.upper_ratio * nUnlabeled
            candidateLabels(sortIndices(floor(parameters.upper_ratio*nUnlabeled)+1:end, dim), dim) = 0;
        elseif sum(currentLabels) < parameters.lower_ratio * nUnlabeled
            candidateLabels(sortIndices(1:ceil(parameters.lower_ratio*nUnlabeled), dim), dim) = 1;
        end
    end
    
    candidateLabels = 2*candidateLabels - 1;
    fullLabelMatrix = [repmat([sourceLabels; targetLabels], [1, 2*embeddingDimension]); candidateLabels];
    
    % Select most violated constraint
    constraintValues = abs(sum([weightedEmbeddings, -weightedEmbeddings] .* fullLabelMatrix));
    [~, bestConstraint] = max(constraintValues);
    selectedLabels = fullLabelMatrix(:, bestConstraint);

    % Update transformation vectors
    weightedSelectedLabels = selectedLabels .* dualCoefficients;
    kernelProjection = kernelRoot * weightedSelectedLabels;
    newDirection = sqrt(parameters.sigma) * kernelProjection / sqrt(kernelProjection' * kernelProjection);
    
    transformationVectors = [transformationVectors newDirection];
    labelMatrix = [labelMatrix selectedLabels];
end

%% Compute Final Decision Threshold
decisionThreshold = combinedKernel * dualCoefficients;
decisionThreshold = mean(decisionThreshold(dualCoefficients > 0));
end