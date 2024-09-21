function [Wx, Wy] = trainSCM_seq(X, Y, nDim, L, Wx, Wy)
    % [Wx, Wy] = trainSCM_seq(X, Y, nDim, L)
    % Computes the Semantic Correlation Maximization orthogonal projection
    % on two data matrices X and Y.
    %
    % Input:
    % X - data matrix of modality x
    % Y - data matrix of modality y
    % nDim - number of SCM-orth dimensions
    % L - label matrix
    % Wx - (optional) default value of projection matrix for modality x
    % Wy - (optional) default value of projection matrix for modality y
    %
    % Output:
    % Wx - SCM-seq projection matrix for modality X
    % Wy - SCM-seq projection matrix for modality Y

    nData = size(X, 1);
    dX = size(X, 2);
    dY = size(Y, 2);
    lambda = 1e-6;

    L(L <= 0) = 0;
    L(L > 0) = 1;

    % normalize the label matrix
    Length = sqrt(sum(L.^2, 2));
    Length(Length == 0) = 1e-8; % avoid division by zero problem for unlabeled rows
    Lambda = 1 ./ Length;
    L = diag(sparse(Lambda)) * L;

    % computes CCA projections if not inputed
    if (~exist('Wx', 'var') || ~exist('Wy', 'var'))
        Cxx = X' * X + 1e-6 * eye(dX);
        Cyy = Y' * Y + 1e-6 * eye(dY);
        Cxy = X' * Y;
        [Wx, Wy, eigv] = trainCCA(Cxx, Cyy, Cxy, nDim);
        Wx = Wx(:, nDim:-1:1);
        Wy = Wy(:, nDim:-1:1);
    end

    CxyB = 2 * (X' * L) * (Y' * L)' - (X' * ones(nData, 1)) * (Y' * ones(nData, 1))';
    Cxy = (nDim * 1) * CxyB;

    Cxx = X' * X + lambda * eye(dX);
    Cyy = Y' * Y + lambda * eye(dY);

    last_eigv = inf;

    for i = 1:nDim
        [wx, wy, eigv] = trainCCA(Cxx, Cyy, Cxy, 1);
        %fprintf('bit%d eigv%f\n', i, eigv);
        if (eigv > last_eigv) % Rarely happens in last bits, if it can't decrease the objective function, use CCA directly.
            %fprintf('use CCA directly\n');
            break;
        end

        last_eigv = eigv;
        Wx(:, i) = wx;
        Wy(:, i) = wy;
        vx = sign(X * wx);
        vy = sign(Y * wy);
        Cxy = Cxy - (X' * vx) * (Y' * vy)';
    end

end
