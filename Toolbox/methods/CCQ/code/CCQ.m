function [model, Bx, By] = CCQ(X, Y, n0, d, m, k, lambda, iters)

    obj = Inf;

    model.type = 'CCQ';
    X = single(X);
    Y = single(Y);
    [px, nx] = size(X);
    [py, ny] = size(Y);
    model.px = px;
    model.py = py;
    model.nx = nx;
    model.ny = ny;
    model.n0 = n0;
    model.d = d;
    model.m = m;

    if length(k) == 1
        k = ones(m, 1) * k;
    end

    bits = sum(log2(k));
    model.k = k;
    model.bits = bits;
    model.lambda = lambda;

    if ~exist('iters', 'var') || isempty(iters)
        iters = 100;
    end

    model.iters = iters;

    Rx = eye(px, d, 'single');
    Ry = eye(py, d, 'single');
    RxX = Rx' * X;
    RyY = Ry' * Y;

    Cx = cell(1, m);
    Cy = cell(1, m);

    for i = 1:m
        perm = randperm(nx, k(i));
        Cx{i} = RxX(:, perm);
        perm = randperm(ny, k(i));
        Cy{i} = RyY(:, perm);
    end

    Bx = zeros(nx, m, 'int32');
    By = zeros(ny, m, 'int32');
    CxBx = zeros([d, nx], 'single');
    CyBy = zeros([d, ny], 'single');

    for i = 1:m
        Bx(:, i) = ccq_nn(Cx{i}, RxX);
        By(:, i) = ccq_nn(Cy{i}, RyY);
        CxBx = CxBx + Cx{i}(:, Bx(:, i));
        CyBy = CyBy + Cy{i}(:, By(:, i));
    end

    for iter = 0:iters

        if (mod(iter, 10) == 0)
            objlast = obj;
            obj = mean(sum((X - Rx * CxBx).^2)) + lambda * mean(sum((Y - Ry * CyBy).^2));
            %fprintf('%3d  %f\n', iter, obj);
            model.obj(iter + 1) = obj;
        end

        if objlast - obj < model.obj(1) * 1e-12
            fprintf('algorithm converged!\n')
            break;
        end

        [Ux, ~, Vx] = svd(X * CxBx', 0);
        [Uy, ~, Vy] = svd(Y * CyBy', 0);
        Rx = Ux * Vx';
        Ry = Uy * Vy';
        RxX = Rx' * X;
        RyY = Ry' * Y;

        eyek = speye(k(1));
        Bx2 = [];

        for i = 1:m
            Bx2 = [Bx2, eyek(Bx(:, i), :)];
        end

        By2 = [];

        for i = 1:m
            By2 = [By2, eyek(By(:, i), :)];
        end

        C2 = (double(RxX) * Bx2 + lambda * double(RyY) * By2) / (Bx2' * Bx2 + lambda * (By2' * By2) + 1e-3 * eye(sum(k)));
        C0 = mat2cell(single(C2), d, k);
        Cx = C0;
        Cy = C0;

        eRxX = RxX;
        eRyY = RyY;
        CxBx = zeros([d, nx], 'single');
        CyBy = zeros([d, ny], 'single');

        for i = 1:m
            Bx(n0 + 1:end, i) = ccq_nn(Cx{i}, eRxX(:, n0 + 1:end));
            By(n0 + 1:end, i) = ccq_nn(Cy{i}, eRyY(:, n0 + 1:end));
            B0 = ccq_nn(Cx{i}, eRxX(:, 1:n0), Cy{i}, eRyY(:, 1:n0), lambda);
            Bx(1:n0, i) = B0;
            By(1:n0, i) = B0;

            eRxX = eRxX - Cx{i}(:, Bx(:, i));
            eRyY = eRyY - Cy{i}(:, By(:, i));
            CxBx = CxBx + Cx{i}(:, Bx(:, i));
            CyBy = CyBy + Cy{i}(:, By(:, i));
        end

    end

    for i = 1:m
        model.Cx{i} = Cx{i};
        model.Cy{i} = Cy{i};
    end

    % Alternative solution for Rx and Ry
    % for better out-of-sample extension
    % Could be commented out and use Lines 71-72

    %% mod by zhangzhen
    Rx = (X * X' + 1e-3 * eye(px)) \ (X * CxBx');
    Ry = (Y * Y' + 1e-3 * eye(py)) \ (Y * CyBy');
    model.Rx = Rx;
    model.Ry = Ry;

    Bx = Bx';
    By = By';

end
