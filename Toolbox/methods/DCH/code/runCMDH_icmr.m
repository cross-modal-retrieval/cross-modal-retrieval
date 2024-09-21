%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% runCMDH_icmr
% Xing Xu
% Limu, Kyushu University, Japan

% This script is to perform the cross-modal discrete hashing (CMDH)
% on three bench mark dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function runCMDH_icmr
%     clear;
%     clc;
      
%     dataset_name = {'nuswide_icmr'};
%     dataset_name = {'mirflickr25k'};
    dataset_name = {'nuswide_icmr'};
    bits = [16, 32, 64, 128];
    runTimes = 1;
    save_path = './exp_icmr';
    
%     diary 'D:\NutstoneCloud\cdh_relax_nuswide.log';
%     diary on;
    % load labelme data
%     load('./voc_data/labelme_sigir14_zhou.mat');
%     load('wikiData.mat');
%     load('mirflickr25k.mat');
%     load('nuswide_lssh.mat');
    
    for d = 1:length(dataset_name)
        dataset = dataset_name{d};
        load(sprintf('../data/%s.mat', dataset));
        
        if strcmp(dataset, 'nuswide_lssh') || strcmp(dataset, 'nuswide_icmr')
            I_tr = [I_tr; I_db];
            T_tr= [T_tr; T_db];
            L_tr = [L_tr; L_db];
        end
        
        if strcmp(dataset, 'wikiData')
            nanchor = 0;
        else
            nanchor = 1000;
        end
                
        % make the training/test data zero-mean
        I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
        I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
        T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
        T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));
        
        fprintf('.. run CMDH on %s dataset: \n', dataset);
        
        for b = 1:length(bits)
            nbits = bits(b);
            for r = 1:runTimes                         
%                [~, ~, MAP_I2T(d,b,r), MAP_T2I(d,b,r)] = ...
%                    SDH_train2_nohashing(I_tr', T_tr', I_te', T_te', L_tr, L_te, nanchor, nbits, true);    
               [~, ~, MAP_I2T(d,b,r), MAP_T2I(d,b,r)] = ...
                   SDH_train2(I_tr', T_tr', I_te', T_te', L_tr, L_te, nanchor, nbits, true);  
            end 
            
            avg_map_i2t = mean(MAP_I2T(d,b,:));
            avg_map_t2i = mean(MAP_T2I(d,b,:));
            
            fprintf('%s: %d bits, avg i2t: %f, avg t2i: %f. \n', dataset, nbits, avg_map_i2t, avg_map_t2i);            
        end
          
    end
    
%     diary off;
    % save data
%     data_name = fullfile(save_path, 'cmdh_3datasets.mat');
%     save(data_name, 'MAP_I2T', 'MAP_T2I');

    fprintf('finished! \n'); 
        
end



