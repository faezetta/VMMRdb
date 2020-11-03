function run = MIL_Test_Validate(data_file, classifier)

global preprocess;
clear run;

% The statistics of dataset
%[X, Y, num_data, num_feature] = Preprocessing(D);
%num_class = length(preprocess.ClassSet);
%class_set = preprocess.ClassSet;
[bags, num_data, num_feature] = MIL_Data_Load(data_file);

% Extract the training and testing data
% X_test = X;
% Y_test = Y;
testindex = 1:num_data;

% Classify with Ensemble 
[test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = MIL_Classify(classifier, [], bags);      
%run.Y_compute = Y_compute; run.Y_prob = Y_prob; run.Y_test = Y_test;
run.bag_label = test_bag_label;
run.inst_label = test_inst_label; 
run.bag_prob = test_bag_prob;
run.inst_prob = test_inst_prob;

% Aggregate the predictions in a shot
%if (preprocess.ShotAvailable == 1), [Y_compute, Y_prob, Y_test] = AggregatePredByShot(Y_compute, Y_prob, Y_test, testindex); end;  

  
% Report the performance
% [run.YY, run.YN, run.NY, run.NN, run.Prec, run.Rec, run.F1, run.Err] = CalculatePerformance(Y_compute, Y_test, class_set);
% if ((preprocess.ComputeMAP == 1) && (length(preprocess.OrgClassSet) == 2)),
%       TrueYprob = Y_prob .* (Y_compute == 1)  + (1 - Y_prob) .* (Y_compute ~= 1);
%       run.AvgPrec = ComputeAP(TrueYprob, Y_test, class_set);
%       run.BaseAvgPrec = ComputeRandAP(Y_test, class_set);       
%       fprintf('AP:%f, Base:%f\n', run.AvgPrec, run.BaseAvgPrec);
% end;    
run.BagAccu = MIL_Bag_Evaluate(bags(testindex), test_bag_label);
if ~isempty(test_inst_label)
    run.InstAccu = MIL_Inst_Evaluate(bags(testindex), test_inst_label);
end;

% run.Y_pred = zeros(length(testindex), 4);
% run.Y_pred(:, 1) = (1:num_data)';
% run.Y_pred(:, 2) = run.Y_prob; 
% run.Y_pred(:, 3) = run.Y_compute; 
% run.Y_pred(:, 4) = run.Y_test;
run.bag_pred = zeros(length(testindex), 3);
run.bag_pred(:, 1) = (1:length(testindex))';
run.bag_pred(:, 2) = run.bag_prob; 
run.bag_pred(:, 3) = run.bag_label; 
run.bag_pred(:, 4) = [bags(testindex).label]';
