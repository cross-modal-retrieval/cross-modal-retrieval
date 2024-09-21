function [Q, R] = ccq_encode(model, X, Y)

    if any(model.k > 256)
        error('uint8 indices required.');
    end

    X = single(X);
    Y = single(Y);

    if isempty(X) && isempty(Y)
        Q = [];
        return;

    elseif ~isempty(X) && isempty(Y)
        m = model.m;
        n = size(X, 2);
        Q = zeros(m, n, 'uint8');

        RxX = model.Rx' * X;
        R = RxX;

        eRxX = RxX;

        for i = 1:m
            B(:, i) = ccq_nn(model.Cx{i}, eRxX);
            eRxX = eRxX - model.Cx{i}(:, B(:, i));
        end

        for i = 1:m
            Q(i, :) = uint8(B(:, i)' - 1);
        end

    elseif isempty(X) && ~isempty(Y)
        m = model.m;
        n = size(Y, 2);
        Q = zeros(m, n, 'uint8');

        RyY = model.Ry' * Y;
        R = RyY;

        eRyY = RyY;

        for i = 1:m
            B(:, i) = ccq_nn(model.Cy{i}, eRyY);
            eRyY = eRyY - model.Cy{i}(:, B(:, i));
        end

        for i = 1:m
            Q(i, :) = uint8(B(:, i)' - 1);
        end

    elseif ~isempty(X) && ~isempty(Y)
        m = model.m;
        n = size(X, 2);
        Q = zeros(m, n, 'uint8');

        RxX = model.Rx' * X;
        RyY = model.Ry' * Y;
        R = (RxX + model.lambda * RyY) / 2;

        eRxX = RxX;
        eRyY = RyY;

        for i = 1:m
            B(:, i) = ccq_nn(model.Cx{i}, eRxX, model.Cy{i}, eRyY, model.lambda);
            eRxX = eRxX - model.Cx{i}(:, B(:, i));
            eRyY = eRyY - model.Cy{i}(:, B(:, i));
        end

        for i = 1:m
            Q(i, :) = uint8(B(:, i)' - 1);
        end

    end

end
