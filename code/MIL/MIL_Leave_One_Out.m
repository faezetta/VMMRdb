% Input pararmeter:
% data_file: data file, including the feature data and output class

function run = MIL_Leave_One_Out(data_file, classifier_wrapper_handle, classifier)

global preprocess;
[bags, num_data, num_feature] = MIL_Data_Load(data_file);

run.bag_pred = zeros(num_data, 3);
run.bag_pred(:, 1) = (1:num_data)';

for i = 1:num_data
    fprintf('Iteration %d ......\n', i);
    % Generate the data indeces for the testing data
    testindex = i;
    trainindex = setdiff(1:num_data, testindex);

    % Classificaiton
    run_class(i) = feval(classifier_wrapper_handle, bags, trainindex, testindex, classifier);

    run.bag_pred(testindex, 2) = run_class(i).bag_prob;
    run.bag_pred(testindex, 3) = run_class(i).bag_label;
    run.bag_pred(testindex, 4) = [bags(testindex).label]';
end

if (isfield(run_class(1), 'BagAccu')), run.BagAccu = mean([run_class(:).BagAccu]); end;
if (isfield(run_class(1), 'InstAccu')), run.BagAccu = mean([run_class(:).InstAccu]); end;

if (isfield(preprocess, 'EnforceDistrib') && preprocess.EnforceDistrib == 1)
   num_pos = 0;
   for i = 1:num_data, num_pos = num_pos + bags(i).label; end;
   [sort_ret, sort_idx ] = sort(run.bag_pred(:,2));
   threshold = sort_ret(num_data - num_pos + 1);   
   run.bag_pred(:, 3) = (run.bag_pred(:,2) >= threshold);
   run.BagAccu = sum(run.bag_pred(:,3) == run.bag_pred(:,4)) / num_data;
end