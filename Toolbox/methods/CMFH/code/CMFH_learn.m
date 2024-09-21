function [model, B] = CMFH_learn(X1, X2, param, bits)
    %SOLVECMFH Summary of this function goes here
    % Collective Matrix Factorization Hashing Algorithm (Unsupervised
    %   minimize_{W1, W2, U1, U2, Y} -sum_{t = 1,2} alphas(t)
    %       * (||X^{t} - YU^{t}||_{F}^{2} + mu * ||Y - X^{t}W^{t}||_{F}^{2})
    %       + gamma * (||U^{t}||_{F}^{2} + ||W^{t}||_{F}^{2} + ||V||_{F}^{2}))
    % Notation:
    % X1: data matrix of View1, each row is a sample vector
    % X2: data matrix of View2, each row is a sample vector
    % alphas: trade off between different views
    % mu: trade off between collective matrix factorization and linean
    % projection
    % gamma: parameter to control the model complexity
    %
    % Reference:
    % GG Ding, Yuchen Guo, Jile Zhou, Jianmin Wang, Philip S. Yu
    % "Collective Matrix Factorization Hashing for Scalable Cross-modality Retrieval"
    % IEEE Transaction on Knowledge and Data Engineering (TKDE)
    % (Manuscript)
    %
    % Version2.0 -- July/2015
    % Written by Yuchen Guo (yuchen.w.guo@gmail.com)
    %
    %
    alphas = param.alphas;
    mu = param.mu;
    gamma = param.gamma;
    %% random initialization
    [row, col] = size(X1); %row Ϊ�����������colΪά��
    [~, colt] = size(X2);
    Y = rand(row, bits); %V' in the paper
    U1 = rand(bits, col); %U1' in the paper
    U2 = rand(bits, colt); %U2' in the paper
    W1 = rand(col, bits); % P1 'in the paper
    W2 = rand(colt, bits); %P2 'in the paper
    threshold = 0.01;
    lastF = 99999999;
    iter = 1;
    %% compute iteratively
    while (true)
        % update U1 and U2
        U1 = (Y' * Y + gamma * eye(bits)) \ Y' * X1;
        U2 = (Y' * Y + gamma * eye(bits)) \ Y' * X2;

        % update Y
        Y = (alphas(1) * X1 * (U1' + mu * W1) + alphas(2) * X2 * (U2' + mu * W2)) / (alphas(1) * (U1 * U1' + mu * eye(bits) + gamma * eye(bits)) + alphas(2) * (U2 * U2' + mu * eye(bits) + gamma * eye(bits)));

        %update W1 and W2
        W1 = mu * (mu * X1' * X1 + gamma * eye(col)) \ X1' * Y;
        W2 = mu * (mu * X2' * X2 + gamma * eye(colt)) \ X2' * Y;

        % compute objective function
        norm1 = norm(X1 - Y * U1, 'fro')^2; %F������ƽ��
        norm2 = norm(X2 - Y * U2, 'fro')^2;
        norm3 = norm(Y - X1 * W1, 'fro')^2;
        norm4 = norm(Y - X2 * W2, 'fro')^2;
        norm5 = gamma * (alphas(1) * ((norm(U1, 'fro')^2 + norm(W1, 'fro')^2) + alphas(2) * (norm(U2, 'fro')^2 + norm(W2, 'fro')^2) + sum(alphas) * norm(Y, 'fro')^2));
        %currentF= alphas(1) * norm1 + alphas(2) * norm2 + alphas(1) * mu * norm3 + alphas(2) * mu * norm4 + norm5;
        currentF = alphas(1) * norm1 + alphas(2) * norm2 + mu * norm3 + mu * norm4 + norm5;
        %fprintf('\nobj at iteration %d: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n', ...
        %   iter, currentF, alphas(1) * norm1 + alphas(2) * norm2, norm3 + norm4, norm5);

        model.U1 = U1;
        model.U2 = U2;
        model.W1 = W1;
        model.W2 = W2;
        B = Y;

        %if (lastF - currentF) < threshold
        if iter == 25
            %fprintf('algorithm converges...\n');
            %fprintf('final obj: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n', ...
            %   currentF, alphas(1) * norm1 + alphas(2) * norm2, norm3 + norm4, norm5);
            return;
        end

        iter = iter + 1;
        lastF = currentF;
    end

end
