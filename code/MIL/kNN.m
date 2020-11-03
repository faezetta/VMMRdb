function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = kNN(para, train_bags, test_bags)

global preprocess;
p = char(ParseParameter(para, {'-BagDistType';'-InstDistType';'-RefNum'; '-CiterRank'}, {'min';'euclidean';'2';'4'}));

bag_dist_type = p(1,:);
inst_dist_type = p(2,:);
num_ref = str2num(p(3,:));
rank_citer = str2num(p(4,:));

if strcmp(bag_dist_type, 'max') && strcmp(bag_dist_type, 'min')
    error('Distance type must be either max or min in the kNN method for MIL!');    
end;

if strcmp(inst_dist_type, 'euclidean') && strcmp(inst_dist_type, 'cosine')
    error('Distance type must be either max or min in the kNN method for MIL!');    
end;

%In the testing-only mode, the training file is missing, we get the model
%(in this lazy training case, just the training examples) from the model file
if isempty(train_bags)
    train_bags = MIL_Data_Load(preprocess.model_file);
elseif (~isempty(preprocess.model_file))    
    MIL_Data_Save(preprocess.model_file, train_bags);
end

num_train_bag = length(train_bags);
num_test_bag  = length(test_bags);

if rank_citer > num_train_bag || rank_citer < 0 || num_ref > num_train_bag || num_ref < 1
    fprintf('num_ref and rank_citer must be smaller than the number of training bags and positive (rank_citer = 0 means not using citer)');
    return;
end;

if strcmp(inst_dist_type, 'cosine')
    for i=1:num_train_bag
        train_bags(i).inst_norm = sum((train_bags(i).instance .^ 2), 2) .^ (1/2);        
        
        zero_idx = full(train_bags(i).inst_norm == 0);
        if any(zero_idx)
            train_bags(i).inst_norm = train_bags(i).inst_norm + zero_idx .* 1e-10;
        end
    end    
    for i=1:num_test_bag
        test_bags(i).inst_norm = sum((test_bags(i).instance .^ 2),2) .^ (1/2);
        
        zero_idx = full(test_bags(i).inst_norm == 0);
        if any(zero_idx)
            test_bags(i).inst_norm = test_bags(i).inst_norm + zero_idx .* 1e-10;
        end
    end    
end

%compute the distances between any two training bags, when citers are used
if rank_citer > 0
    for i=1:num_train_bag
        for j= i+1 : num_train_bag
            dist_matrix(i,j) = HausdorffDist(train_bags(i),train_bags(j));            
            dist_matrix(j,i) = dist_matrix(i,j);
        end
        dist_matrix(i,i) = intmax;
    end    
end

%predict the label for each testing bag
for i = 1:num_test_bag    
    bag_dist = zeros(num_train_bag, 1);
    
    select_label = [];
    for j = 1:num_train_bag
        bag_dist(j) = HausdorffDist(test_bags(i), train_bags(j), bag_dist_type, inst_dist_type);        
    end;
    [sort_dist, sort_idx] = sort(bag_dist);
    for j = 1 : min([num_ref num_train_bag])
        select_label(j) = train_bags(sort_idx(j)).label;        
    end
    
    if rank_citer > 0
        idx = min([num_ref num_train_bag]) + 1;
        for j=1:num_train_bag
            [sort_dist, sort_idx] = sort(dist_matrix(j,:));                         
            if bag_dist(j) < sort_dist(rank_citer)
                select_label(idx) = train_bags(j).label;
                idx = idx + 1;
            end;
        end;
    end;
    
    num_pos_label = sum(select_label == 1);
    num_neg_label = sum(select_label == 0);
    
    test_bag_label(i) = (num_pos_label >= num_neg_label);
    test_bag_prob(i) = (num_pos_label / (num_pos_label + num_neg_label));
end;

test_inst_label = [];
test_inst_prob = [];


function dist = HausdorffDist(bag_A, bag_B, type, metric)

if nargin < 4, metric = 'euclidean'; end;
if nargin < 3, type = 'min'; end;

num_A = size(bag_A.instance, 1);
num_B = size(bag_B.instance, 1);

%compute pair-wise distance
inst_dist = zeros(num_A, num_B);
for i = 1:num_A
    for j = 1:num_B
        if strcmp(metric, 'cosine')            
            inst_dist(i,j) = sum(bag_A.instance(i,:) .* bag_B.instance(j,:)) / (bag_A.inst_norm(i) * bag_B.inst_norm(j));
            if inst_dist(i,j) == 0
                inst_dist(i,j) = intmax;
            else
                inst_dist(i,j) = 1 / inst_dist(i,j);
            end
        else
            inst_dist(i,j) = sum((bag_A.instance(i,:) - bag_B.instance(j,:)).^2);
        end
    end        
end

dist_AB = min(inst_dist,[],2);
dist_BA = min(inst_dist,[],1);

if strcmp(type, 'max')
    dist = max(max(dist_AB), max(dist_BA));
else
    dist = max(min(dist_AB), min(dist_BA));
end
