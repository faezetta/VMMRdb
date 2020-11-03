% Input pararmeter: 
% data_file: data file, including the feature data and output class

function run = MIL_Cross_Validate(data_file, classifier_wrapper_handle, classifier)

global preprocess;
%[X, Y, num_data, num_feature] = preprocessing(D);
%clear D;
[bags, num_data, num_feature] = MIL_Data_Load(data_file);

% The statistics of dataset
num_folder = preprocess.NumCrossFolder;
%num_class = length(preprocess.ClassSet);
%class_set = preprocess.ClassSet;

% run.Y_pred = zeros(num_data, 4);
% run.Y_pred(:, 1) = (1:num_data)';
  run.bag_pred = zeros(num_data, 3);
  run.bag_pred(:, 1) = (1:num_data)';

for i = 1:num_folder
  fprintf('Iteration %d ......\n', i);  
  % Generate the data indeces for the testing data
  testindex = floor((i-1) * num_data / num_folder)+1 : floor( i * num_data/num_folder);
  
%   if (preprocess.ShotAvailable == 1) & (preprocess.ValidateByShot == 1)      
%     num_shot = length(preprocess.ShotIDSet);
%     ValidateTestShot = preprocess.ShotIDSet(floor((i-1) * num_shot / num_folder) + 1 : floor(i * num_shot / num_folder));
%     testindex = []; for j = 1:length(ValidateTestShot), testindex = [testindex; find(preprocess.ShotInfo == ValidateTestShot(j))]; end;
%   end;  
  
  trainindex = setdiff(1:num_data, testindex);
  
  % Classificaiton
  run_class(i) = feval(classifier_wrapper_handle, bags, trainindex, testindex, classifier); 
  
  run.bag_pred(testindex, 2) = run_class(i).bag_prob; 
  run.bag_pred(testindex, 3) = run_class(i).bag_label; 
  run.bag_pred(testindex, 4) = [bags(testindex).label]';  
  
%   run.Y_pred(testindex, 2) = run_class(i).Y_prob; 
%   run.Y_pred(testindex, 3) = run_class(i).Y_compute; 
%   run.Y_pred(testindex, 4) = run_class(i).Y_test;
end

% if (isfield(run_class(1), 'Err')), run.Err = mean([run_class(:).Err]); end;
% if (isfield(run_class(1), 'Prec')), run.Prec = mean([run_class(:).Prec]); end;
% if (isfield(run_class(1), 'Rec')), run.Rec = mean([run_class(:).Rec]); end;
% if (isfield(run_class(1), 'F1')), run.F1 = mean([run_class(:).F1]); end;
% if (isfield(run_class(1), 'Micro_Prec')), run.Micro_Prec = mean([run_class(:).Micro_Prec]); end;
% if (isfield(run_class(1), 'Micro_Rec')), run.Micro_Rec = mean([run_class(:).Micro_Rec]); end;
% if (isfield(run_class(1), 'Micro_F1')), run.Micro_F1 = mean([run_class(:).Micro_F1]); end;
% if (isfield(run_class(1), 'Macro_Prec')), run.Macro_Prec = mean([run_class(:).Macro_Prec]); end;
% if (isfield(run_class(1), 'Macro_Rec')), run.Macro_Rec = mean([run_class(:).Macro_Rec]); end;
% if (isfield(run_class(1), 'Macro_F1')), run.Macro_F1 = mean([run_class(:).Macro_F1]); end;
% if (isfield(run_class(1), 'AvgPrec')), run.AvgPrec = mean([run_class(:).AvgPrec]); end;
% if (isfield(run_class(1), 'BaseAvgPrec')), run.BaseAvgPrec = mean([run_class(:).BaseAvgPrec]); end;

if (isfield(run_class(1), 'BagAccu')), run.BagAccu = mean([run_class(:).BagAccu]); end;
if (isfield(run_class(1), 'InstAccu')), run.InstAccu = mean([run_class(:).InstAccu]); end;

if (isfield(preprocess, 'EnforceDistrib') && preprocess.EnforceDistrib == 1)
   num_pos = 0;
   for i = 1:num_data, num_pos = num_pos + bags(i).label; end;
   [sort_ret, sort_idx ] = sort(run.bag_pred(:,2));
   threshold = sort_ret(num_data - num_pos + 1);   
   run.bag_pred(:, 3) = (run.bag_pred(:,2) >= threshold);
   run.BagAccu = sum(run.bag_pred(:,3) == run.bag_pred(:,4)) / num_data;
end

% function RemoveConstraints()
% 
% global preprocess;
% if (preprocess.ConstraintAvailable == 1) & (preprocess.ShotAvailable == 1)
%       for j = 1:size(preprocess.constraintMap, 1),
%           ShotInfo = preprocess.ShotInfo;
%           preprocess.constraintUsed(j) = (all(ShotInfo(trainindex) ~= preprocess.constraintMap(j,1)) && ...
%               all(ShotInfo(trainindex) ~= preprocess.constraintMap(j,2)));
%       end;
% end;

