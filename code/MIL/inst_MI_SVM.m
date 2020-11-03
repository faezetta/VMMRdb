function [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = inst_MI_SVM(para, train_bags, test_bags)

global preprocess;
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

num_train_bag = length(train_bags);
num_test_bag = length(test_bags);

%set the initial instance labels to bag labels
idx = 0;
for i=1:num_train_bag
    num_inst = size(train_bags(i).instance, 1);
    train_label(idx+1 : idx+num_inst) = repmat(train_bags(i).label, num_inst, 1);    
    idx = idx + num_inst;
end

[train_instance, dummy] = bag2instance(train_bags);
[test_instance, dummy] = bag2instance(test_bags);

num_train_inst = size(train_instance, 1);
num_test_inst = size(test_instance, 1);

if isempty(train_bags)
    if (~isfield(preprocess, 'model_file') || isempty(preprocess.model_file))
        error('The model file must be provided in the train_only setting!');
    end;
    eval(['!copy ' preprocess.model_file ' ' temp_model_file ]);
    [test_label_predict, test_prob_predict] = LibSVM(para, [], [], test_instance, ones(num_test_inst, 1));    
else
    
    step = 1;
    past_train_label(step,:) = train_label;
    
    while 1
        %num_pos_label = sum(train_label == 1);
        %num_neg_label = sum(train_label == 0);
        %new_para = sprintf(' -NegativeWeight %.10g', (num_pos_label / num_neg_label));
        
        [all_label_predict, all_prob_predict] = LibSVM(para, train_instance, train_label, [train_instance; test_instance], ones(num_train_inst+num_test_inst, 1));
        train_label_predict = all_label_predict(1 : num_train_inst);
        train_prob_predict = all_prob_predict(1 : num_train_inst);
        test_label_predict = all_label_predict(num_train_inst+1 : num_train_inst+ num_test_inst);
        test_prob_predict = all_prob_predict(num_train_inst+1 : num_train_inst+ num_test_inst);

        idx = 0;
        for i=1:num_train_bag
            num_inst = size(train_bags(i).instance, 1);

            if train_bags(i).label == 0
                train_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
            else
                if any(train_label_predict(idx+1 : idx+num_inst))
                    train_label(idx+1 : idx+num_inst) = train_label_predict(idx+1 : idx+num_inst);
                else
                    [sort_prob, sort_idx] = sort(train_prob_predict(idx+1 : idx+num_inst));
                    train_label(idx+1 : idx+num_inst) = zeros(num_inst, 1);
                    train_label(idx + sort_idx(num_inst)) = 1;
                end
            end
            idx = idx + num_inst;
        end
        
        difference = sum(past_train_label(step,:) ~= train_label);
        fprintf('Number of label changes is %d\n', difference);
        if difference == 0, break; end;
         
        repeat_label = 0;
        for i = 1 : step
            if all(train_label == past_train_label(i, :))
                repeat_label = 1;
                break;
            end               
        end

        if repeat_label == 1
            fprintf('Repeated training labels found, quit...\n');
            break; 
        end

        step = step + 1;
        past_train_label(step, :) = train_label;
         
    end    
end

%prediction
test_inst_label = test_label_predict;
test_inst_prob = test_prob_predict;

idx = 0;
test_bag_label = zeros(num_test_bag, 1);
for i=1:num_test_bag
    num_inst = size(test_bags(i).instance, 1);    
    test_bag_label(i) = any(test_inst_label(idx+1 : idx+num_inst));
    test_bag_prob(i) = max(test_inst_prob(idx+1 : idx+num_inst));
    idx = idx + num_inst;
end