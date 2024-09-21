function map = mAP(sim_x,L_tr,L_te,mark)
%sim_x(i,j) denote the sim bewteen query j and database i 
%
% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Jile Zhou (zhoujile539@gmail.com)
%

tn = size(sim_x,2);
APx = zeros(tn,1);
R = 100;
tmp_mat = L_te*L_tr';%���?
label_mat = tmp_mat>=1;%����label_mat������Ԫ�ص�tmp_mat���ڵ���1ʱΪ1������Ϊ0
%L_tr = [L_tr;L_tr];
for i = 1 : tn
    Px = zeros(R,1);
    deltax = zeros(R,1);
%     label = L_te(i);
    if mark == 0
        [~,inxx] = sort(sim_x(:,i),'descend');
    elseif mark == 1
        [~,inxx] = sort(sim_x(:,i));
    end
    Lx = length(find(label_mat(i, inxx(1:R)) == 1));
    for r = 1 : R
        Lrx = length(find(label_mat(i, inxx(1:r)) == 1));
        if 1 == label_mat(i, inxx(r))
            deltax(r) = 1;
        end
        Px(r) = Lrx/r;
    end
    if Lx ~=0
        APx(i) = sum(Px.*deltax)/Lx;
    end
end
map = mean(APx);

