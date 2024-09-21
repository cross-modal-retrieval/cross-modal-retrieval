function [code_trainX, code_trainY, code_test_X, code_test_T] = CMFH_compress1(model, I_db, T_db, I_te, T_te)

    W1 = model.W1;
    W2 = model.W2;

    code_trainX = I_db * W1;
    %code_trainX= sign((bsxfun(@minus, code_trainX , mean(code_trainX,1))));% 2173��bits
    code_trainX = code_trainX > 0;

    code_trainY = T_db * W2;
    %code_trainY= sign((bsxfun(@minus, code_trainY , mean(code_trainY,1))));% 2173��bits
    code_trainY = code_trainY > 0;
    % traintime=toc;

    code_test_X = sign(I_te * W1); %((bsxfun(@minus,I_te * W1 , mean(Y,1)))); %693��bits
    code_test_X = code_test_X > 0;
    code_test_T = sign(T_te * W2); %((bsxfun(@minus,T_te * W2 , mean(Y,1))));
    code_test_T = code_test_T > 0;

end
