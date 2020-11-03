function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = EMDD(para, train_bags, test_bags)

global preprocess;
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

p = char(ParseParameter(para, {'-Scaling';'-NumRuns';'-Aggregate';'-Threshold';'-IterTolerance'}, {'1';'10';'avg';'0.5';'0.1'}));

scale_var = str2num(p(1,:));
num_runs = str2num(p(2,:));
aggregate = p(3,:);
threshold = str2num(p(4,:));
tolerance = str2num(p(5,:));

if isempty(train_bags)
    %in testing only setting, load the APR data from the model file
    if (~isfield(preprocess, 'model_file') || isempty(preprocess.model_file))
        error('The model file name must be provided in the test-only setting');
    end;
    load('-mat', preprocess.model_file, 'target', 'scale', 'fval');
else
    [target, scale, fval] = EMDD_train(train_bags, tolerance, scale_var, num_runs);
    save(temp_model_file, 'target', 'scale', 'fval');
end;

[test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = DD_predict(target, scale, fval, test_bags, aggregate, threshold);

function [target, scale, fval] = EMDD_train(bags, fval_tol, scale_var, nruns)
% if nargin < 4, nruns = 10; end;
% if nargin < 3, scale_var = 1; end;
% if nargin < 2, fval_tol = 0.1; end;

nbag = length(bags);
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

    [t, s, f] = run_EMDD(bags, fval_tol, scale_var, bags(bag_idx).instance(inst_idx,:), ones(1, nfea));
    
    target(idx, :) = t;
    scale(idx, :) = s;
    fval(idx) = f;    
end 

function [target, scale, final_fval] = run_EMDD(bags, fval_tol, scale_var, init_target, init_scale)
nbag = length(bags);
target = init_target;
scale = init_scale;

% select an optimal instance from each bag according to 
% the current target and scale, saved into sel_insts
fval_diff = inf;
prev_fval = inf;
final_fval = 0;

while fval_diff > fval_tol
    for i=1:nbag
        insts = bags(i).instance;
        [ninst, nfea] = size(insts);
        
        t = repmat(target, ninst, 1);
        s = repmat(scale, ninst, 1);
        
        dist = mean((((insts - t).^2) .* (s.^2)),2);
        bags(i).inst_prob = exp(-dist);
        
        [ret, idx] = sort(bags(i).inst_prob);
        
        sel_insts(i, :) = insts(idx(ninst), :);    
        sel_labels(i) = bags(i).label;
    end
    
    if scale_var == 1
        init_params = [target, scale];           
        lb = zeros(1, 2*nfea);
        ub = [ones(1, nfea), repmat(inf, 1, nfea)];
%       eq_left = [zeros(1, nfea), ones(1, nfea)];
%       eq_right = nfea;               
%       ineq_left = [zeros(1, nfea), -ones(1, nfea)];
%       ineq_right = - nfea;       
        [params, fv] = fmincon(@density_func, init_params, [], [], [], [], lb, ub, [], optimset('Display', 'iter', 'GradObj', 'off', 'LargeScale', 'on', 'MaxFunEvals', 50000, 'MaxIter', 1000, 'TolFun', 1.0e-03, 'TolX', 1.0e-010, 'TolCon', 1.0e-010), sel_insts, sel_labels );
        
        target = params(1 : nfea);
        scale = params(nfea+1 : 2*nfea);
    else
        init_params = target;
        lb = zeros(1, nfea);
        ub = ones(1, nfea);
        
        [params, fv] = fmincon(@density_func, init_params, [], [], [], [], lb, ub, [], optimset('Display', 'iter', 'GradObj', 'on', 'LargeScale', 'on', 'MaxFunEvals', 50000, 'MaxIter', 1000, 'TolFun', 1.0e-03, 'TolX', 1.0e-010, 'TolCon', 1.0e-010), sel_insts, sel_labels);
        
        target = params;           
        scale  = ones(1, nfea);
    end     
    final_fval = fv;
    fval_diff = prev_fval - final_fval;
    prev_fval = final_fval;
end

function [bag_label, inst_label, bag_prob, inst_prob] = DD_predict(target, scale, fval, bags, aggregate, threshold)
[num_bag, num_inst, num_feature] = MIL_Size(bags);
num_run = length(fval);
bag_label = zeros(num_bag, 1);
inst_label = zeros(num_inst, 1);
inst_prob = zeros(num_inst, 1);

nbag = length(bags);
if strcmp(aggregate,'max') || strcmp(aggregate,'min')
    
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
        bag_prob(i) = max(inst_prob(idx +1 : idx + ninst));
        bag_label(i) = any(inst_label(idx +1 : idx + ninst));
                   
        idx = idx + ninst;
    end    

elseif strcmp(aggregate,'avg')

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
        bag_prob(i) = max(inst_prob(idx +1 : idx + ninst));
        bag_label(i) = any(inst_label(idx +1 : idx + ninst));
        
        idx = idx + ninst;
    end
else
    error('The aggregate must be max, min, or avg!');
end

function [fun, grad] = density_func(params, insts, labels)
[ninst, nfea] = size(insts);

if length(params) == nfea
    target = params;           
    scale = ones(1, nfea);
    scale_var = 0;
else
    target = params(1 : nfea);
    scale = params(nfea+1 : 2*nfea);    
    scale_var = 1;
end

t = repmat(target, ninst, 1);
s = repmat(scale, ninst, 1);
    
dist = mean((((insts - t).^2) .* (s.^2)),2);
prob = exp(-dist);

fun = 0;
for i=1:ninst
    if labels(i) == 1
        if prob(i) == 0, prob(i) = 1.0e-10; end
        fun = fun - log(prob(i));
    else
        if prob(i) == 1, prob(i) = 1- (1.0e-10); end
        fun = fun - log(1-prob(i)); 
    end;
end

%calculate the gradient
if nargout > 1  % fun called with two outputs including the gradient
    if scale_var == 1,   grad = zeros(1, nfea * 2);
    else, grad = zeros(1, nfea); end;

    for d = 1:nfea
        for i =1:ninst
            if labels(i) == 1
                grad(d) = grad(d) - (2/nfea) * (scale(d)^2) * (insts(i, d) - target(d));                
                if scale_var == 1, grad(d + nfea) = grad(d + nfea) + (2/nfea) * scale(d) * ((insts(i, d)-target(d)) ^ 2);  end;
            else
                grad(d) = grad(d) + (1/(1-prob(i))) * (2/nfea) * (scale(d)^2) * (insts(i, d) - target(d));
                if scale_var == 1, grad(d + nfea) = grad(d + nfea) - (1/(1-prob(i))) * (2/nfea) * scale(d) * ((insts(i,d)-target(d)) ^ 2); end;
            end
        end
    end
end