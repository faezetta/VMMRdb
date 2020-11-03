function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = DD(para, train_bags, test_bags)

global preprocess;
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

p = char(ParseParameter(para, {'-Scaling';'-NumRuns';'-Aggregate';'-Threshold'}, {'1';'10';'avg';'0.5'}));

scale_var = str2num(p(1,:));
num_runs = str2num(p(2,:));
aggregate = p(3,:);
threshold = str2num(p(4,:));

if isempty(train_bags)
    %in testing only setting, load the APR data from the model file
    if (~isfield(preprocess, 'model_file') || isempty(preprocess.model_file))
        error('The model file name must be provided in the test-only setting');
    end;
    load('-mat', preprocess.model_file, 'target', 'scale', 'fval');
else
    [target, scale, fval] = DD_train(train_bags, scale_var, num_runs);
    save(temp_model_file, 'target', 'scale', 'fval');    
end;

[test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = DD_predict(target, scale, fval, test_bags, aggregate, threshold);

function [target, scale, fval] = DD_train(bags, scale_var, nruns)

nbag = length(bags);

%initilaize the signals showing whether an instances has been used as
%starting point to 0 (false)
for i = 1:nbag
    ninst = size(bags(i).instance, 1);
    bags(i).starting_point = zeros(ninst, 1);
end

for idx = 1:nruns
    %randomly choose a positive bag
    bag_idx = ceil(rand(1) * nbag);    
    while (bags(bag_idx).label == 0) || (all(bags(bag_idx).starting_point) == 1)
        bag_idx = ceil(rand(1) * nbag);
    end
    
    %randomly choose an instnace from the bag that is not used before
    [ninst, nfea] = size(bags(bag_idx).instance);    
    inst_idx = ceil(rand(1) * ninst);    
    while  bags(bag_idx).starting_point(inst_idx) == 1
          inst_idx = ceil(rand(1) * ninst);    
    end
    bags(bag_idx).starting_point(inst_idx) = 1;
    
    if scale_var == 1                
        init_params = [bags(bag_idx).instance(inst_idx,:), repmat(1, 1, nfea)];  
        
        lb = [zeros(1, nfea), repmat(0, 1, nfea)];
        ub = [ones(1, nfea), repmat(inf, 1, nfea)];
        eq_left = [zeros(1, nfea), ones(1, nfea)];
        eq_right = nfea;
        ineq_left = [zeros(1, nfea), -ones(1, nfea)];
        ineq_right = - nfea;

        [params, fv] = fmincon(@dd_func, init_params, [], [], [], [], lb, ub, [], optimset('Display', 'iter', 'GradObj', 'off', 'LargeScale', 'off', 'MaxFunEvals', 100000, 'MaxIter', 1000, 'TolFun', 1.0e-3, 'TolX', 1.0e-010, 'TolCon', 1.0e-010), bags);

        target(idx, :) = params(1 : nfea);
        scale(idx, :) = params(nfea+1 : 2*nfea);
    else
        init_params = bags(i).instance(ceil(rand(1)*ninst),:);
        lb = zeros(1, nfea);
        ub = ones(1, nfea);

        [params, fv] = fmincon(@dd_func, init_params, [], [], [], [], lb, ub, [], optimset('Display', 'iter', 'GradObj', 'off', 'LargeScale', 'off', 'MaxFunEvals', 50000, 'MaxIter', 1000, 'TolFun', 1.0e-10, 'TolX', 1.0e-010, 'TolCon', 1.0e-010), bags);

        target(idx, :) = params;
        scale(idx, :) = ones(1, nfea);
    end
    fval(idx) = fv;    
end

function [bag_label, inst_label, bag_prob, inst_prob] = DD_predict(target, scale, fval, bags, aggregate, threshold)
[num_bag, num_inst, num_feature] = MIL_Size(bags);
num_run = length(fval);
bag_label = zeros(num_bag, 1);
inst_label = zeros(num_inst, 1);
inst_prob = zeros(num_inst, 1);

nbag = length(bags);
if strcmp(aggregate, 'max') || strcmp(aggregate, 'min')
    
    [f, fidx] = sort(fval);
    if strcmp(aggregate, 'max')
        t = target(fidx(num_run), :);
        s = scale(fidx(num_run), :);    
    else
        t = target(fidx(1), :);
        s = scale(fidx(1), :);    
    end

    idx = 0;
    for i = 1:nbag        
        [ninst, nfea] = size(bags(i).instance);    
        t_mat = repmat(t, ninst, 1);
        s_mat = repmat(s, ninst, 1);
    
        dist = mean((((bags(i).instance - t_mat).^2) .* (s_mat.^2)),2);
        inst_prob(idx+1 : idx+ninst) = exp(-dist);
        inst_label(idx +1 : idx + ninst) = (inst_prob(idx+1 : idx+ninst) >= repmat(threshold, ninst, 1));
        bag_prob(i) = max(inst_prob(idx+1 : idx+ninst));
        bag_label(i) = any(inst_label(idx +1 : idx + ninst));
                   
        idx = idx + ninst;
    end    

elseif strcmp(aggregate, 'avg')

    idx = 0; 
    for i = 1:nbag
        [ninst, nfea] = size(bags(i).instance);
                  
        for j = 1:num_run
            t_mat = repmat(target(j,:), ninst, 1);
            s_mat = repmat(scale(j,:), ninst, 1);    
            dist = mean((((bags(i).instance - t_mat).^2) .* (s_mat.^2)),2);
            inst_prob(idx+1 : idx+ninst) = inst_prob(idx+1 : idx+ninst) + exp(-dist);
        end

        inst_prob(idx+1 : idx+ninst)  = inst_prob(idx+1 : idx+ninst) ./ num_run;
        inst_label(idx +1 : idx + ninst) = (inst_prob(idx+1 : idx+ninst) >= repmat(threshold, ninst, 1));
        bag_prob(i) = max(inst_prob(idx+1 : idx+ninst));
        bag_label(i) = any(inst_label(idx +1 : idx + ninst));
        
        idx = idx + ninst;
    end
else
    error('The aggregate must be max, min, or avg!');
end

function [fun, grad] = dd_func(params, bags)
fun = 0;
nbag = length(bags);
nfea = size(bags(1).instance, 2);

if length(params) == nfea
    target = params;           
    scale = ones(1, nfea);
    scale_var = 0;
else
    target = params(1 : nfea);
    scale = params(nfea+1 : 2*nfea);    
    scale_var = 1;
end

for i = 1:nbag    
    insts = bags(i).instance;
    ninst = size(insts, 1);
    
    t = repmat(target, ninst, 1);
    s = repmat(scale, ninst, 1);
    
    dist = mean((((insts - t).^2) .* (s.^2)),2);
    bags(i).inst_prob = exp(-dist);
    bags(i).prob = 1 - prod(1 - bags(i).inst_prob);  
    
    if bags(i).label == 1
        if bags(i).prob == 0, bags(i).prob = 1.0e-10; end;
        fun = fun - log(bags(i).prob);
    else
        if bags(i).prob == 1
            bags(i).prob = 1 - (1.0e-10);
        end
        fun = fun - log(1-bags(i).prob);
    end
end

%calculate the gradient
if nargout > 1  % fun called with two outputs including the gradient
    if scale_var == 1,   grad = zeros(1, nfea * 2);
    else, grad = zeros(1, nfea); end;

    for d = 1:nfea
        for i =1:nbag                           
            if all(bags(i).inst_prob ~= 1) == 0     %handle numerical difficulty  that denominator is equal to 0
                test = (bags(i).inst_prob == 1);
                bags(i).inst_prob = bags(i).inst_prob - test .* 1.0e-5;                
            end           

            if bags(i).label == 1
                %gradient for target(d) and scale(d) for postivie bags
                grad(d) = grad(d) - 2 * (((1/bags(i).prob) - 1) * sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) .* (scale(d).^ 2) .* (bags(i).instance(:,d) - target(d))' ./ nfea));

                if scale_var == 1
                    grad(d+nfea) = grad(d+nfea) + 2 * ((1/bags(i).prob) - 1) * sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) .* scale(d) .* (((bags(i).instance(:,d) - target(d))') .^ 2) ./ nfea);            
                end
            else
               %gradient for target(d) and scale(d) for negative bags
                grad(d) = grad(d) + 2 * sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) .* (scale(d).^ 2) .* (bags(i).instance(:,d) - target(d))' ./ nfea);            
                
                if scale_var == 1
                    grad(d+nfea) = grad(d+nfea) - 2 * sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) * scale(d) .* (((bags(i).instance(:,d) - target(d))') .^ 2) ./ nfea);                            
                end
            end            
        end
    end
end

% function fun = normalized_dd_func(params, bags)
% fun = 0;
% nbag = length(bags);
% nfea = size(bags(1).instance, 2);
% 
% if length(params) == nfea
%     target = params;           
%     scale = ones(1, nfea);
%     scale_var = 0;
% else
%     target = params(1 : nfea);
%     scale = params(nfea+1 : 2*nfea);    
%     scale_var = 1;
% end
% 
% for i = 1:nbag    
%     insts = bags(i).instance;
%     ninst = size(insts, 1);
%     
%     t_mat = repmat(target, ninst, 1);
%     s_mat = repmat(scale, ninst, 1);
%     
%     if prod(scale) == 0
%       bags(i).inst_prob = 0;
%       bags(i).prob = 0;
%     else
%         normalizer = 1/(((2*pi)^(nfea/2)) * prod(scale));  
%         dist = sum((((insts - t_mat).^2) ./ (s_mat.^2)),2);
%         bags(i).inst_prob = normalizer .* exp((-0.5) .* dist);
%         bags(i).prob = normalizer * (1- prod(1 - bags(i).inst_prob ./ normalizer));  
%     end
%     
%     if bags(i).label == 1        
%         if bags(i).prob == 0, bags(i).prob = 1.0e-10; end
%         fun = fun - log(bags(i).prob);
%     else
%         fun = fun - log(normalizer) - sum(log(1 - bags(i).inst_prob ./ normalizer));
%     end
% end
% if fun < -10000
%     x = 1;
% end
% %calculate the gradient
% % if nargout > 1  % fun called with two outputs including the gradient
% %     if scale_var == 1,   grad = zeros(1, nfea * 2);
% %     else, grad = zeros(1, nfea); end;
% % 
% %     for d = 1:nfea
% %         for i =1:nbag           
% %                 
% %             if all(bags(i).inst_prob ~= 1) == 0     %handle numerical difficulty  that denominator is equal to 0
% %                 test = (bags(i).inst_prob == 1);
% %                 bags(i).inst_prob = bags(i).inst_prob - test .* 1.0e-5;                
% %             end           
% % 
% %             if bags(i).label == 1
% %                 %gradient for target(d) 
% %                 grad(d) = grad(d) - ((1/bags(i).prob) - 1) * sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) .* (1/(((2*pi)^(nfea/2)) * prod(scale))) .* (1/ scale(d)^2) .* (bags(i).instance(:,d) - target(d))');
% % 
% %                 %gradient for scale(d)
% %                 if scale_var == 1
% %                     grad(d+nfea) = grad(d+nfea) - ((1/bags(i).prob) - 1) * sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) .* (((((bags(i).instance(:,d) - target(d))').^ 2) ./ (scale(d)^3)) - (1/scale(d))));            
% %                 end
% %             else
% %                %gradient for target(d) 
% %                 grad(d) = grad(d) + sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) .* (1/(((2*pi)^(nfea/2)) * prod(scale))) .* (1/scale(d)^2) .* (bags(i).instance(:,d) - target(d))'); 
% %                 
% %                 %gradient for scale(d)
% %                 if scale_var == 1
% %                     grad(d+nfea) = grad(d+nfea) + sum(((bags(i).inst_prob) ./ (1 - bags(i).inst_prob)) * (((((bags(i).instance(:,d) - target(d))').^ 2) ./ (scale(d)^3)) - (1/scale(d))));
% %                 end
% %             end            
% %         end
% %     end
% % end

