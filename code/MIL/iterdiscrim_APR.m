function  [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = iterdiscrim_APR(para, train_bags, test_bags)

global preprocess;
p = str2num(char(ParseParameter(para, {'-NegMargin'; '-GridNum'; '-InsideProb'; '-OutsideProb'}, {'100';'10000';'0.999';'0.01'})));

margin = p(1);
num_grid = p(2);
inside_prob = p(3);
outside_prob = p(4);

[num_train_bag, num_train_inst, num_feature] = MIL_Size(train_bags);
[num_test_bag, num_test_inst, num_feature] = MIL_Size(test_bags);

if isempty(train_bags)
    %in testing only setting, load the APR data from the model file
    if (~isfield(preprocess, 'model_file') || isempty(preprocess.model_file))
        error('The model file name must be provided in the test-only setting');
    end;

    %load the APR data if model_filename is provided
    fid = fopen(preprocess.model_file, 'r');
    if fid == -1, error('model file cannot be opened for reading!'); end;
    fclose(fid);
    load('-mat', preprocess.model_file, 'feature_range', 'lb', 'ub');
else
    %if training data is providde, we train the APR
    minmax_lb = repmat(intmax, 1, num_feature);
    minmax_ub = repmat(-intmax, 1, num_feature);
    for i = 1:num_train_bag
        if train_bags(i).label == 0, continue; end;
        for k = 1 : num_feature
            minmax_lb(k) = min(max(train_bags(i).instance(:,k)), minmax_lb(k));
            minmax_ub(k) = max(min(train_bags(i).instance(:,k)), minmax_ub(k));
        end;
    end;

    %set the initial positive instance to the one closest to the minmax APR
    idx = 0;
    dist = repmat(intmax, num_train_inst, 1);
    for i = 1:num_train_bag
        if train_bags(i).label == 0, continue; end;
        num_inst = size(train_bags(i).instance, 1);
        for j = 1 : num_inst
            dist(idx + j) = dist2APR(minmax_lb, minmax_ub, train_bags(i).instance(j,:));
        end;
        idx = idx + num_inst;
    end;
    [sort_ret, sort_idx] = sort(dist);
    idx = 0;
    for i = 1 : num_train_bag
        if train_bags(i).label == 0, continue; end;
        num_inst = size(train_bags(i).instance, 1);
        if (idx + num_inst) >= sort_idx(1), break; end;
        idx = idx + num_inst;
    end
    init_bag_choice = i;
    init_inst_choice = sort_idx(1) - idx;
    init_inst = train_bags(i).instance(init_inst_choice,:);

    %start training
    feature_range = ones(1, num_feature);
    while 1

        ub = init_inst;
        lb = init_inst;

        %expand an APR tighest covers at least one instance per positive bag
        pos_covered = zeros(num_train_bag, 1);
        for i = 1 : num_train_bag
            pos_covered(i) = ~train_bags(i).label;
        end
        pos_covered(init_bag_choice) = 1;
        pos_bag_choice(1) = init_bag_choice;
        pos_bag_repinst(1) = init_inst_choice;

        step = 2;
        while  all(pos_covered) == 0
            size_increase = zeros(num_train_inst, 1);

            idx = 0;
            for i = 1 : num_train_bag
                [num_inst, num_feature] = size(train_bags(i).instance);
                
                 %skip if it is a negative bag or a covered positive bag
                if train_bags(i).label == 0 || pos_covered(i) == 1
                    size_increase(idx+1 : idx+num_inst) = repmat(intmax, num_inst, 1);
                    idx = idx + num_inst;
                    continue; 
                end

                for j = 1 : num_inst
                    for k = 1 : num_feature
                        if feature_range(k) == 0, continue; end;
                        if train_bags(i).instance(j,k) < lb(k)
                            size_increase(idx+j) =  size_increase(idx+j) + (lb(k) - train_bags(i).instance(j,k));
                        elseif train_bags(i).instance(j,k) > ub(k)
                            size_increase(idx+j) =  size_increase(idx+j) + (train_bags(i).instance(j,k) - ub(k));
                        end
                    end
                end
                idx = idx + num_inst;
            end

            %select the positive instance that increases the APR the least
            [sort_size, sort_idx] = sort(size_increase);
            choice = sort_idx(1);

            idx = 0;
            for i = 1 : num_train_bag
                num_inst = size(train_bags(i).instance, 1);
                if (idx + num_inst) >= choice, break; end;
                idx = idx + num_inst;
            end
            choice_inst = train_bags(i).instance((choice - idx), :)';
            pos_covered(i) = 1;
            pos_bag_choice(step) = i;
            pos_bag_repinst(step) = choice - idx;

            %expand the APR to include this closest node
            for k = 1 : length(choice_inst)
                if feature_range(k) == 0, continue; end;
                if choice_inst(k) < lb(k)
                    lb(k) = choice_inst(k);
                elseif choice_inst(k) > ub(k)
                    ub(k) = choice_inst(k);
                end
            end

            %backfitting to see if any instance in covered positive bags can be switched to get a tighter APR
            clear pos_inst_choice;
            while 1
                prev_pos_bag_repinst = pos_bag_repinst;
                for i = 1 : step                    
                    pos_bag_repinst(i) = adjust_APR(train_bags, pos_bag_choice, pos_bag_repinst, feature_range, i);
                    pos_inst_choice(i, :) = train_bags(pos_bag_choice(i)).instance(pos_bag_repinst(i), :);
                end
                [lb, ub] = find_bounding_APR(pos_inst_choice, feature_range);
                if(prev_pos_bag_repinst == pos_bag_repinst)
                    break;
                else
                    bbb = 1;
                end
            end
            step = step + 1;
        end

        %feature selection       
        idx = 0;
        neg_excluded = zeros(num_train_inst, 1);
        for i = 1 : num_train_bag            
            num_inst = size(train_bags(i).instance, 1);
            neg_excluded(idx+1 : idx + num_inst) = repmat(train_bags(i).label, num_inst, 1);
            idx = idx + num_inst;
        end

        %calcuate a matrix showing how much each instance is outside the APR
        neg_in_APR = 0;
        dist_APR = zeros(num_train_inst, num_feature);
        idx = 0;
        for i = 1 : num_train_bag
            num_inst = size(train_bags(i).instance, 1);
            for j = 1 : num_inst
                for k = 1 : num_feature
                    if feature_range(k) == 0, continue; end;
                    if train_bags(i).instance(j, k) >= lb(k) && train_bags(i).instance(j, k) <= ub(k)
                        dist_APR(idx + j, k) = 0;
                    else
                        if train_bags(i).instance(j, k) < lb(k), dist_APR(idx + j, k) = lb(k) - train_bags(i).instance(j, k);
                        else dist_APR(idx + j, k) = train_bags(i).instance(j, k) - ub(k); end;
                    end
                end
                if all(dist_APR(idx + j, :) == 0) && train_bags(i).label == 0
                    fprintf('a negative instance  is in the APR');
                    neg_in_APR = 1;
                    break;
                end
            end
            idx = idx + num_inst;
        end

        %if a negative instance in the current APR, there is no need to do
        %feature selection since no feature can help remove this instance
        if(neg_in_APR == 1), break; end;
        
        feature_selected = ~feature_range; %suppose removed features are already selected so that they don't need to be considered
        
        %select features iteratively that exclude most neg examples,
        %until all neg instances are removed or all features are selected
        while all(neg_excluded) == 0 && all(feature_selected) == 0
            num_neg_against = zeros(num_feature, 1);
            
            %compute the # of negative instances this feature against
            for k = 1 : num_feature     
                %skip selected features and removed features
                if feature_range(k) == 0 || feature_selected(k) == 1, continue;  end;
                idx = 0;
                for i = 1 : num_train_bag
                    num_inst = size(train_bags(i).instance, 1);
                    for j = 1 : num_inst
                        if neg_excluded(idx + j) == 1, continue; end;  %either positive instance or excluded negative instance
                        if dist_APR(idx + j, k) > margin || (dist_APR(idx+j, k) > 0 && dist_APR(idx+j, k) == max(dist_APR(idx+j,:)))
                            num_neg_against(k) = num_neg_against(k) + 1;
                        end
                    end
                    idx = idx + num_inst;
                end
            end

            %find the most discriminative feature
            [sort_ret, sort_idx] = sort(num_neg_against);
            choice = sort_idx(length(sort_ret));
            feature_selected(choice) = 1;

            %update the list of remaining neg instances
            idx = 0;
            for i = 1 : num_train_bag
                num_inst = size(train_bags(i).instance, 1);
                for j = 1 : num_inst
                    if neg_excluded(idx + j) == 1, continue; end;  %either positive instance or excluded negative instance
                    if dist_APR(idx + j, choice) > margin || (dist_APR(idx+j, choice) > 0 && dist_APR(idx+j, choice) == max(dist_APR(idx+j,:)))
                        neg_excluded(idx+j) = 1;
                    end
                end
                idx = idx + num_inst;
            end
        end

        %if all feature are selected, the algorithm converges and returns
        if all(feature_selected),  break;  end;

        %update feature_range according to feature_selected
        for k = 1 : num_feature
            if feature_range(k) == 1 && feature_selected(k) == 0
                feature_range(k) = 0;
            end
        end
    end

    fprintf('%d expands to ',APR_size(lb, ub, feature_range)); 
    %expanding the APR using kernel density estimation
    overall_lb = zeros(1, num_feature);
    overall_ub = zeros(1, num_feature);
    for k = 1 : num_feature;
        for i = 1 : num_train_bag
            overall_lb(k) = min(min(train_bags(i).instance(:,k)), overall_lb(k));
            overall_ub(k) = max(max(train_bags(i).instance(:,k)), overall_ub(k));
        end
    end
    overall_lb = overall_lb - 50;
    overall_ub = overall_ub + 50;
    grid_size = (overall_ub - overall_lb) ./ num_grid;

    for k = 1 : num_feature;
        idx = 1;
        if feature_range(k) == 0, continue; end;
        
        %choose the instances falling into the current APR for kernal estimation
        for i = 1 : num_train_bag
            if train_bags(i).label == 0, continue; end;
            for j = 1 : size(train_bags(i).instance, 1)
                 if train_bags(i).instance(j,k) >= lb(k) && train_bags(i).instance(j,k) <= ub(k)
                    value(idx) = train_bags(i).instance(j, k);
                    idx = idx + 1;
                end
            end
        end
        grid = overall_lb(k) : grid_size(k) : overall_ub(k);
        
        %calculate kernel width, which makes the inside probability with
        %the current APR equal to the given inside probability 
        if inside_prob == 0.999
            kernel_width = (ub(k) - lb(k)) / (2 * 3.291);
        elseif inside_prob == 0.995
            kernel_width = (ub(k) - lb(k)) / (2 * 2.807);
        elseif inside_prob == 0.99
            kernel_width = (ub(k) - lb(k)) / (2 * 2.576);
        elseif inside_prob == 0.95
            kernel_width = (ub(k) - lb(k)) / (2 * 1.960);
        else
            wrong = 1;
        end
        
        ds = ksdensity(value, grid, 'width', kernel_width);

        %expand the APR according to the kernel estimation
        accum_prob = 0;
        lb_set = 0;
        for i = 1 : length(ds) - 1
            accum_prob = accum_prob + grid_size(k) * ((ds(i) + ds(i+1))/2);
            if accum_prob > (outside_prob/2) && lb_set == 0
                if lb(k) > grid(i), lb(k) = grid(i); end;
                lb_set = 1;
            end;
            if accum_prob > (1 - outside_prob/2)
                if ub(k) < grid(i), ub(k) = grid(i); end;
                break;
            end;
        end
    end
    fprintf('%d.\n',APR_size(lb, ub, feature_range)); 

    if (isfield(preprocess,'model_file') && ~isempty(preprocess.model_file))
        %save the APR data if model_filename is provided
        fid = fopen(preprocess.model_file, 'w');
        if fid == -1, error('model file cannot be opened for writing!'); end;
        fclose(fid);
        save(preprocess.model_file, 'feature_range', 'lb', 'ub');
    end;
end;

%prediction
idx = 0;
test_bag_label = zeros(num_test_bag, 1);
test_inst_label = zeros(num_test_inst, 1);
test_inst_prob = [];
for i = 1 : num_test_bag
    num_inst = size(test_bags(i).instance, 1);
    for j = 1 : num_inst
        test_inst_label(idx + j) = (dist2APR(lb, ub, test_bags(i).instance(j, :), feature_range) == 0);
    end
    test_bag_prob(i) = sum(test_inst_label(idx+1 : idx+num_inst)) / num_inst;
    test_bag_label(i) = any(test_inst_label(idx+1 : idx+num_inst));
    idx = idx + num_inst;
end

function inst_choice = adjust_APR(bags, pos_bag_choice, pos_bag_repinst, feature_range, revisit_step)
num_step = length(pos_bag_choice);
idx = 1;
for i = 1 : num_step
    if i ~= revisit_step
        select_inst(idx, :) = bags(pos_bag_choice(i)).instance(pos_bag_repinst(i), :);
        idx = idx + 1;
    end
end

[lb, ub] = find_bounding_APR(select_inst, feature_range);
base_size = APR_size(lb, ub, feature_range);

num_inst_revisit = size(bags(pos_bag_choice(revisit_step)).instance, 1);
for i = 1 : num_inst_revisit
    revisit_inst = bags(pos_bag_choice(revisit_step)).instance(i, :);
    [lb, ub] = find_bounding_APR([select_inst; revisit_inst], feature_range);
    size_increase(i) = APR_size(lb, ub, feature_range) - base_size;
end

[sort_ret, sort_idx] = sort(size_increase);
inst_choice = sort_idx(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lb, ub] = find_bounding_APR(inst, feature_range)
[num_inst, num_feature] = size(inst);
if nargin < 2, feature_range = ones(1, num_feature); end;

ub = zeros(1, num_feature);
lb = zeros(1, num_feature);
for i = 1 : num_feature
    if feature_range(i)
        ub(i) = max(inst(:, i));
        lb(i) = min(inst(:, i));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function size = APR_size(lb, ub, feature_range)
num_feature = length(ub);
if nargin < 3, feature_range = ones(1, num_feature); end;
size = sum((ub - lb) .* feature_range);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dist = dist2APR(lb, ub, inst, feature_range)
num_feature = length(ub);
if nargin < 4, feature_range = ones(1, num_feature); end;
dist = 0;
for i = 1:num_feature
    if feature_range(i) == 0, continue; end;
    if inst(i) < lb(i)
        dist = dist + (lb(i) - inst(i));
    elseif inst(i) > ub(i)
        dist = dist + (inst(i)  - ub(i)); end;
end
