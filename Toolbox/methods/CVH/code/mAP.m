function map = mAP(sim_x,L_tr,L_te,mark)

tn = size(sim_x,2);
APx = zeros(tn,1);
R = 100;
tmp_mat = L_te*L_tr';
label_mat = tmp_mat>=1;

for i = 1 : tn
    Px = zeros(R,1);
    deltax = zeros(R,1);
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

