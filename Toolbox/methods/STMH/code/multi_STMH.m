function [outG, outFCell, R, D, outObj, outNumIter] = multi_STMH(train_dataX, train_dataT, inPara, numzeros)
    % parameter settings
    maxIter = inPara.maxIter;
    thresh = inPara.thresh;
    r = inPara.r;
    lamda = inPara.lamda;
    gamma = inPara.gamma;
    bits = inPara.bits;
    numdata = size(train_dataX, 2);
    % inti common indicator G
    G{1} = randn(numdata, bits);
    % initialize G
    G{2} = zeros(numdata, bits);

    for i = 1:numdata
        index = randperm(bits);
        G{2}(i, index(1:numzeros)) = 1;
    end

    D = sparse(1:numdata, 1:numdata, ones(numdata, 1));
    obj = zeros(maxIter, 1);
    % loop
    for t = 1:maxIter
        %fprintf('processing iteration %d...\n', t);

        % update F{v}
        M = (G{1}' * D * G{1}) + (gamma / r(1)) * eye(bits);
        N = train_dataX * D * G{1};
        F{1} = N / M;
        DF = G{2}' * G{2} + (gamma / r(1)) * eye(bits);
        %DF =  diag(sum(G{2},1)) + (gamma/r(1))*eye(bits);
        F{2} = train_dataT * G{2} / DF;

        % undate R
        R = (G{1}' * G{1} + (gamma / lamda) * eye(bits)) \ (G{1}' * G{2});

        % undate G{1}
        %G{1} = inXCell{1}'*F{1}/(F{1}'*F{1} + (gamma/r(1))*eye(bits));
        left = r(1) * D * train_dataX' * F{1} + lamda * G{2} * R';
        right1 = r(1) * F{1}' * F{1};
        right2 = lamda * R * R' + gamma * eye(bits);

        for i = 1:numdata
            G{1}(i, :) = left(i, :) / (D(i, i) * right1 + right2);
        end

        % undate G{2}
        GR = G{1} * R;

        for i = 1:numdata
            xVec = train_dataT(:, i);
            G{2}(i, :) = searchBestIndicator(r(2), xVec, F{2}, GR(i, :), numzeros, lamda);
        end

        % update D{v}
        E = (train_dataX - F{1} * G{1}')';
        EE = sqrt(sum(E .* E, 2) + eps);
        temp = (0.5 ./ EE) * (r(1));
        D = sparse(1:numdata, 1:numdata, temp);

        % calculate the obj
        NO1 = r(1) * sum(EE);
        DX = spdiags(sum(G{2}, 2), 0, numdata, numdata);
        DF = spdiags(sum(G{2}, 1)', 0, bits, bits);
        NO2 = r(2) * (trace(train_dataT * DX * train_dataT' - 2 * train_dataT * G{2} * F{2}' + F{2} * DF * F{2}'));
        NO3 = lamda * norm(G{2} - G{1} * R)^2;
        NO4 = gamma * (norm(R)^2 + norm(G{1})^2 + norm(F{1})^2 + norm(F{2})^2);
        obj(t) = NO1 + NO2 + NO3 + NO4;
        %fprintf('%d...\n',NO1(1),NO1(2),NO2,NO3,obj(t));
        if (t > 1)
            diff = abs(obj(t - 1) - obj(t));

            if (diff < thresh)
                fprintf('algorithm converges...\n');
                break;
            end

        end

    end

    outObj = obj;
    outNumIter = t;
    outFCell = F;
    outG = G;

end

function outVec = searchBestIndicator(dVec, xCell, F, GR, numzeros, lamda)
    % solve the following problem,
    c = size(F, 2);
    tmp = zeros(c, 1);
    obj = zeros(c, 1);

    for j = 1:c
        aa = norm(xCell - F(:, j))^2;
        bb = 2 * GR(j);
        obj(j, 1) = dVec * aa - lamda * bb;
    end

    [~, idx] = sort(obj, 'ascend');
    tmp(idx(1:numzeros)) = 1;
    outVec = tmp;
end
