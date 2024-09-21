function [U1, U2, P1, P2, S] = solveSMFH(X1, X2, L, alpha, beta, gamma, lambda, bits)
    %% Reference:
    % Jun Tang, Ke Wang, Ling Shao
    % "Supervised Matrix Factorization Hashing for Cross-Modal Retrieval"
    % IEEE Transaction on Image Processing (TIP)
    % (Manuscript)
    %
    % Version1.0 -- June/2016
    % Written by Jun Tang(tangjunahu@163.com) and Ke Wang(wangke4747@126.com)
    %
    %

    %% random initialization
    [row1, col1] = size(X1);
    [row2, col2] = size(X2);
    S = rand(bits, col1);
    U1 = rand(row1, bits);
    U2 = rand(row2, bits);
    P1 = rand(bits, row1);
    P2 = rand(bits, row2);
    threshold = 1;
    lastF = 99999999;
    iter = 1;

    %% compute iteratively
    while (true)
        % update U1 and U2
        U1 = X1 * S' / (S * S' + (lambda / alpha) * eye(bits));
        U2 = X2 * S' / (S * S' + (lambda / (1 - alpha)) * eye(bits));

        % update S
        A = 2 * (alpha * U1' * U1 + (1 - alpha) * U2' * U2 + 2 * beta * eye(bits) + lambda * eye(bits));
        B = gamma * (L + L');
        C = -2 * (alpha * U1' * X1 + (1 - alpha) * U2' * X2 + beta * (P1 * X1 + P2 * X2));
        S = lyap(A, B, C);

        %update P1 and P2
        P1 = S * X1' / (X1 * X1' + (lambda / beta) * eye(row1));
        P2 = S * X2' / (X2 * X2' + (lambda / beta) * eye(row2));

        % compute objective function
        norm1 = alpha * (norm(X1 - U1 * S, 'fro')^2);
        norm2 = (1 - alpha) * (norm(X2 - U2 * S, 'fro')^2);
        norm3 = beta * (norm(S - P1 * X1, 'fro')^2);
        norm4 = beta * (norm(S - P2 * X2, 'fro')^2);
        norm5 = gamma * trace(S * L * S');
        norm6 = lambda * (norm(U1, 'fro')^2 + norm(U2, 'fro')^2 + norm(S, 'fro')^2 + norm(P1, 'fro')^2 + norm(P2, 'fro')^2);
        currentF = norm1 + norm2 + norm3 + norm4 + norm5 + norm6;
        %fprintf('\nobj at iteration %d: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for linear projection: %.4f,\n joint graph term: %.4f,\n regularization term: %.4f\n\n', iter, currentF, norm1 + norm2, norm3 + norm4, norm5, norm6);
        if (lastF - currentF) < threshold
            %fprintf('algorithm converges...\n');
            %fprintf('final obj: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for linear projection: %.4f,\n joint graph term: %.4f,\n regularization term: %.4f\n\n', currentF,norm1 + norm2, norm3 + norm4, norm5, norm6);
            return;
        end

        %fprintf('iter %d, loss = %d\n', iter, lastF - currentF);
        iter = iter + 1;
        lastF = currentF;
    end

    return;
end
