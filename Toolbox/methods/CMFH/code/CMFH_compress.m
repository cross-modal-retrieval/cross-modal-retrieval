function [code_train, code_test_X, code_test_T] = CMFH_compress(model, B, I_tr, T_tr, I_te, T_te, param, bits)
    alphas = param.alphas; % lambda in the paper,0.5 in the paper
    mu = param.mu; % mu in the paper
    gamma = param.gamma; % gamma in the paper

    U1 = model.U1;
    U2 = model.U2;
    W1 = model.W1;
    W2 = model.W2;
    Y = B;
    code_train = (alphas(1) * I_tr * (U1' + mu * W1) + alphas(2) * T_tr * (U2' + mu * W2)) / (alphas(1) * (U1 * U1' + mu * eye(bits) + gamma * eye(bits)) + alphas(2) * (U2 * U2' + mu * eye(bits) + gamma * eye(bits)));
    code_train = sign((bsxfun(@minus, code_train, mean(Y, 1)))); % 2173��bits
    code_train = code_train > 0;
    code_test_X = sign(I_te * W1); %((bsxfun(@minus,I_te * W1 , mean(Y,1)))); %693��bits
    code_test_X = code_test_X > 0;
    code_test_T = sign(T_te * W2); %((bsxfun(@minus,T_te * W2 , mean(Y,1))));
    code_test_T = code_test_T > 0;
    code_train = code_train';
    code_test_X = code_test_X';
    code_test_T = code_test_T';
end
