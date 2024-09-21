function map = mAP(sim_x,L_tr,L_te)
%sim_x(i,j) denote the sim bewteen query j and database i
tn = size(sim_x,2);
APx = zeros(tn,1);
R = 2173;

%L_tr = [L_tr;L_tr];
[row col] = size(L_tr);
if min(row,col) == 1
    for i = 1 : tn
        Px = zeros(R,1);
        deltax = zeros(R,1);
        label = L_te(i);
        [tempx,inxx] = sort(sim_x(:,i),'descend');
        Lx = length(find(L_tr(inxx(1:R)) == label));
        for r = 1 : R
            Lrx = length(find(L_tr(inxx(1:r)) == label));
            if label == L_tr(inxx(r))
                deltax(r) = 1;
            end
            Px(r) = Lrx/r;
        end
        if Lx ~=0
            APx(i) = sum(Px.*deltax)/Lx;
        end
    end
    map = mean(APx);
end
    
