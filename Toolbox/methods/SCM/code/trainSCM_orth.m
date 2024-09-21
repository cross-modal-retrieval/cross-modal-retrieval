function [Wx, Wy] = trainSCM_orth(X, Y, nDim, L)
    % [Wx, Wy] = trainSCM_orth(X, Y, nDim, L)
    % Computes the Semantic Correlation Maximization orthogonal projection
    % on two data matrices X and Y.
    %
    % Input:
    % X - data matrix of modality x
    % Y - data matrix of modality y
    % nDim - number of SCM-orth dimensions
    % L - label matrix
    %
    % Output:
    % Wx - SCM-orth projection matrix for modality X
    % Wy - SCM-orth projection matrix for modality Y

    nData = size(X, 1);
    dX = size(X, 2);
    dY = size(Y, 2);
    lambda = 1e-6;

    L(L <= 0) = 0;
    L(L > 0) = 1;

    % normalize the label matrix
    Length = sqrt(sum(L.^2, 2));
    Length(Length <= 0) = 1e-8; % avoid division by zero problem for unlabeled rows
    Lambda = 1 ./ Length;
    L = diag(sparse(Lambda)) * L;

    CxyB = 2 * (X' * L) * (Y' * L)' - (X' * ones(nData, 1)) * (Y' * ones(nData, 1))';
    Cxy = (nDim * 1) * CxyB;

    Cxx = X' * X + lambda * eye(dX);
    Cyy = Y' * Y + lambda * eye(dY);

    [Wx, Wy, eigv] = trainCCA(Cxx, Cyy, Cxy, nDim);

end
