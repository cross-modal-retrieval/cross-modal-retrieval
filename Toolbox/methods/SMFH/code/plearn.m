function labels = plearn (S, plabs)

    %
    % labels = plearn(S,pl)
    %
    % S - NxN symmetric matrix with real values. Usually sparse.
    % pl - Kx1 vector with integer values [1..m]. K < N
    %
    % Returns: labels - (N-K)x1 vector with integer values [1..m]
    %
    % Labels a partially labeled data set using graph regularization
    % with boundary conditions.
    %
    % S is the smoothness matrix for the dataset. That typically would
    % be the Laplacian or Laplacian squared.
    % pl is the set of labels for the first K vectors in the dataset.
    % All labels [1..m] should be present.
    %
    % Note: The data matrix itself is not used by this routine.
    %
    % Example:
    %
    % S = laplacian (X,'nn',6); calculate the Laplacian for the dataset X
    % labels = plearn(S^2,pl)
    % vector labels contains labels for the last N-K data points.
    %
    %
    % Author:
    %
    % Mikhail Belkin
    % misha@math.uchicago.edu
    %

    if (nargin < 2)
        disp(sprintf('ERROR: Too few arguments given.\n'));
    end;

    % number of labeled points
    n = length (plabs);

    % total number of points
    N = size (S, 1);

    L1 = S(n + 1:N, n + 1:N);
    L2 = S(1:n, n + 1:N);

    c1 = min(plabs);
    cn = max(plabs);
    % number of classes
    classes = cn - c1 + 1;

    Y = zeros (classes, n);
    X = zeros (classes, N - n);

    for c = c1:cn

        I = find (plabs == c);
        Y(c, I) = 1;
    end

    Y = L2' * Y';

    tol = 1e-6;
    maxit = 200;

    %[LL,UU] = luinc(L1,1e-2);

    for i = 1:classes
        X(i, :) = -bicgstab(L1, Y(:, i), tol, maxit)';
    end;

    for i = 1:N - n
        ii = find (X(:, i) == max (X(:, i)));
        labels(i) = c1 + min(ii) - 1;
    end;
