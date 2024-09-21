function [Wx, Wy] = CVHtrainCCA(Cxx, Cyy, Cxy, dCCA)
    %
    % Input
    %   Cxx : covariance matrxi of X-X
    %   Cyy : covariance matrxi of Y-Y
    %   Cxy : covariance matrxi of X-Y
    %   dCCA : # of CCA dimensions
    % Output:
    %   Wx  : projection matrix of X
    %   Wy  : projection matrix of Y
    %

    % [Ndimx Ndimy] = size(Cxy);

    % eta = S3PLHparam.eta;
    % magx = diag(X*X');
    % alpha = 1/max(magx);
    % alpha = 0.1;
    % S = S3PLHparam.S;
    % selectedidx = S3PLHparam.selectedidx;
    % XS = X(selectedidx,:);
    % SS = S(selectedidx,selectedidx); clear S;

    % algorithm
    option = struct('disp', 0); %creates a structure array with the specified fields and values.
    % lambda = 1e-6;
    % Cxx = X*X'+lambda*eye(Ndimx);
    % Cyy = Y*Y'+lambda*eye(Ndimy);
    % Cxy = X*Y';
    A = Cxy * (Cyy) * Cxy'; B = Cxx;
    % size(A)
    % size(B)
    [eigvectors eigvalues] = eigs(A, B, dCCA, 'lr', option);
    %[eigvectors eigvalues] = eig(A,B);
    eigvalues = real(eigvalues); eigvectors = real(eigvectors);

    if min(diag(eigvalues)) <= 0
        fprintf('Retrieved eigenvectors wrong!\n');
    end

    % 2) store paramaters
    % sy = size(Y,1);
    Wx = eigvectors;
    Wy = inv(Cyy) * Cxy' * eigvectors * inv(sqrt(eigvalues)); %
    %CMHparam.Wy = CMHparam.Wy./repmat(sqrt(sum(abs(CMHparam.Wy).^2)),sy,1); % Normalize Wy
