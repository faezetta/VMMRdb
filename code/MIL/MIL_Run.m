% MIL_run('classify -t input_file [options] [-- EvaluationMethod [options]] –- BaseClassifier [options]'); 
% MIL_run('classify -t example.data –sf 1 -n 1 -- cross_validate -t 5 -- DD -NumRuns 10 –Aggregate avg');
function run = MIL_Run(classifier)

warning('off','MATLAB:colon:operandsNotRealScalar');

% clear global preprocess;
global preprocess; 
global temp_train_file temp_test_file temp_output_file temp_model_file weka_dir mySVM_dir libSVM_dir SVMLight_dir; 
% preprocess = [];

if (~isfield(preprocess, 'Message')), preprocess.Message = ''; end;
if (~isfield(preprocess, 'NumCrossFolder')), preprocess.NumCrossFolder = 3; end;
if (~isfield(preprocess, 'TrainTestSplitBoundary')), preprocess.TrainTestSplitBoundary = 100; end;
if (~isfield(preprocess, 'Normalization')), preprocess.Normalization = 1; end;
%if (~isfield(preprocess, 'SizeFactor')), preprocess.SizeFactor = 0.5; end;
%if (~isfield(preprocess, 'ShotAvailable')), preprocess.ShotAvailable = 0; end;
%if (~isfield(preprocess, 'DataSampling')), preprocess.DataSampling = 0; end;
%if (~isfield(preprocess, 'Sparse')), preprocess.Sparse = 0; end;
if (~isfield(preprocess, 'Shuffled')), preprocess.Shuffled = 0; end;
if (~isfield(preprocess, 'OutputFlag')), preprocess.OutputFlag = 'a'; end;
%if (~isfield(preprocess, 'SVD')), preprocess.SVD = 0; end;
%if (~isfield(preprocess, 'FLD')), preprocess.FLD = 0; end;
%if (~isfield(preprocess, 'CHI')), preprocess.ChiSquare = 0; end;
%if (~isfield(preprocess, 'ValidateByShot')), preprocess.ValidateByShot = 0; end;
%if (~isfield(preprocess, 'Ensemble')), preprocess.Ensemble = 0; end;
%if (~isfield(preprocess, 'ComputeMAP')), preprocess.ComputeMAP = 0; end;
if (~isfield(preprocess, 'Evaluation')), preprocess.Evaluation = 0; preprocess.TrainTestSplitBoundary = -2; end;
%if (~isfield(preprocess, 'MultiClassType')), preprocess.MultiClassType = 0; end;
%if (~isfield(preprocess, 'MultiClass') | (preprocess.MultiClassType == 0)), 
%    preprocess.MultiClass.LabelType = 1; preprocess.MultiClass.CodeType = -1; preprocess.MultiClass.LossFuncType = -1;
%    preprocess.MultiClass.UncertaintyFuncType = -1; preprocess.MultiClass.ProbEstimation = -1;
%end;
%if (~isfield(preprocess, 'ConstraintAvailable')), preprocess.ConstraintAvailable = 0; end;
%if (~isfield(preprocess, 'ConstraintFileName')), preprocess.ConstraintFileName = ''; end;
if (~isfield(preprocess, 'input_file')), preprocess.input_file = ''; end;
if (~isfield(preprocess, 'output_file')), preprocess.output_file = ''; end;
if (~isfield(preprocess, 'pred_file')), preprocess.pred_file = ''; end;
if (~isfield(preprocess, 'model_file')), preprocess.model_file = ''; end;
if (~isfield(preprocess, 'WorkingDir')), preprocess.WorkingDir = ''; end;
if (~isfield(preprocess, 'EnforceDistrib')), preprocess.EnforceDistrib = 0; end;
if (~isfield(preprocess, 'InputFormat')), preprocess.InputFormat = 0; end;


if (nargin < 1), Report_Error; end;
[header, para, rem] = ParseCmd(classifier, '--');
if (strcmpi(header, 'classify')), 
    p = str2num(char(ParseParameter(para, {'-v'; '-sf'; '-n'; '-if'; '-of'; '-pf'; '-distrib'}, ...
                                          {'1'; '0'; '1'; '0'; '0'; '0'; '0'})));
    preprocess.Vebosity = p(1);
    preprocess.Shuffled = p(2);
    preprocess.Normalization = p(3);
    preprocess.InputFormat = p(4);
    preprocess.OutputFormat = p(5);
    preprocess.PredFormat = p(6);
    preprocess.EnforceDistrib = p(7);    
%   preprocess.ShotAvailable = p(4);
%   preprocess.ValidateByShot = p(5);
%   preprocess.DataSampling = p(6);
%   preprocess.DataSamplingRate = p(7);
%   preprocess.SVD = p(8);
%   preprocess.FLD = p(9);
%   preprocess.ComputeMAP = p(10);
%   preprocess.InputFormat = p(11);
%   preprocess.ChiSquare = p(12);    
%   preprocess.Sparse = p(15);
%   preprocess.EnforceDistrib = p(16);
    
    p = ParseParameter(para, {'-t'; '-o'; '-p'; '-oflag'; '-dir'; '-drf' }, {''; ''; ''; 'a'; ''; ''});
    preprocess.input_file = char(p{1, :});
    preprocess.output_file = char(p{2, :});    
    preprocess.pred_file = char(p{3, :});    
    preprocess.OutputFlag = char(p{4, :});    
    preprocess.WorkingDir = char(p{5, :});    
 %  preprocess.DimReductionFile = char(p{6, :});    
    classifier = rem;   
else 
    Report_Error;
end;

% Setup the environmental varaible for directory information
if (isempty(preprocess.WorkingDir)),     
    filename = 'MIL_Classify.m'; 
    if (~exist(filename)),
        error('Cannot find the files in LibMIL!');
    end;
    cur_dir = which(filename); 
    sep_pos = findstr(cur_dir, filesep); 
    preprocess.WorkingDir = cur_dir(1:sep_pos(length(sep_pos))-1);
end;
root = preprocess.WorkingDir;

temp_dir = sprintf(['%s' filesep 'temp'], root);
if (~exist(temp_dir)),    
    s = mkdir(root, 'temp');
    if (s ~= 1), error('Cannot create temp directory!'); end;
end;
temp_train_file = sprintf(['%s' filesep 'temp.train.txt'], temp_dir);
temp_test_file = sprintf(['%s' filesep 'temp.test.txt'], temp_dir);
temp_output_file = sprintf(['%s' filesep 'temp.output.txt'], temp_dir);
temp_model_file = sprintf(['%s' filesep 'temp.model.txt'], temp_dir);
%weka_dir = sprintf('%s/weka-3-4/weka.jar', root);
%5mySVM_dir = sprintf('%s/svm', root);
libSVM_dir = sprintf(['%s' filesep 'svm'], root);
%SVMLight_dir = sprintf('%s/svm', root);

[header, para, rem] = ParseCmd(classifier, '--');
if (strcmpi(header, 'train_test_validate')), 
    preprocess.Evaluation = 0;
    p = str2num(char(ParseParameter(para, {'-t'}, {'-2'})));
    preprocess.TrainTestSplitBoundary = p(1);
    classifier = rem;
elseif (strcmpi(header, 'cross_validate')), 
    preprocess.Evaluation = 1;
    p = str2num(char(ParseParameter(para, {'-t'}, {'3'})));
    preprocess.NumCrossFolder = p(1);
    classifier = rem;
elseif (strcmpi(header, 'test_file_validate')), 
    preprocess.Evaluation = 2;
    p = char(ParseParameter(para, {'-t'}, {''}));
    preprocess.test_file = p(1, :);
    classifier = rem;
elseif (strcmpi(header, 'train_only')), 
    preprocess.Evaluation = 3;
    p = char(ParseParameter(para, {'-m'}, {''}));
    preprocess.model_file = p(1, :);
    temp_model_file = preprocess.model_file;
    classifier = rem;
elseif (strcmpi(header, 'test_only')), 
    preprocess.Evaluation = 4;
    p = char(ParseParameter(para, {'-m'}, {''}));
    preprocess.model_file = p(1, :);
    temp_model_file = preprocess.model_file;
    classifier = rem;
elseif (strcmpi(header, 'leave_one_out')), 
    preprocess.Evaluation = 5;    
    classifier = rem;
end;   

% [header, para, rem] = ParseCmd(classifier, '--');
% if (strcmpi(header, 'train_test_simple')), 
%     preprocess.MultiClassType = 0;
%     p = str2num(char(ParseParameter(para, {'-LabelType'}, {'1'})));
%     preprocess.MultiClass.LabelType = p(1);
%     classifier = rem;
% elseif (strcmpi(header, 'train_test_multiple_class')), 
%     preprocess.MultiClassType = 1;
%     p = str2num(char(ParseParameter(para, {'-LabelType'; '-CodeType'; '-LossFuncType'}, {'1'; '0'; '2'})));
%     preprocess.MultiClass.LabelType = p(1);
%     preprocess.MultiClass.CodeType = p(2);
%     preprocess.MultiClass.LossFuncType = p(3);
%     preprocess.MultiClass.UncertaintyFuncType = 2; 
%     preprocess.MultiClass.ProbEstimation = 0;
%     classifier = rem;
% elseif (strcmpi(header, 'train_test_multiple_label')), 
%     preprocess.MultiClassType = 2;
%     p = str2num(char(ParseParameter(para, {'-LabelType'}, {'1'})));
%     preprocess.MultiClass.LabelType = p(1);
%     classifier = rem;
% elseif (strcmpi(header, 'train_test_multiple_class_AL')), 
%     preprocess.MultiClassType = 3;
%     p = str2num(char(ParseParameter(para, {'-LabelType'; '-CodeType'; '-LossFuncType'; '-ALIter'; '-ALIncrSize'}, {'1'; '0'; '2'; '4'; '10'})));
%     preprocess.MultiClass.LabelType = p(1);
%     preprocess.MultiClass.CodeType = p(2);F@
%     preprocess.MultiClass.LossFuncType = p(3);
%     preprocess.ActiveLearning.Iteration = p(4);
%     preprocess.ActiveLearning.IncrementSize = p(5);
%     preprocess.MultiClass.UncertaintyFuncType = 2; 
%     preprocess.MultiClass.ProbEstimation = 0;
%     classifier = rem;
% end;   

% Initialize the message string
preprocess.Message = '';
if (preprocess.Evaluation == 0)
    msg = sprintf(' Train-Test Split, Boundary: %d, ', preprocess.TrainTestSplitBoundary);
    preprocess.Message = [preprocess.Message msg]; 
elseif (preprocess.Evaluation == 1)
    msg = sprintf(' Cross Validation, Folder: %d, ', preprocess.NumCrossFolder);
    preprocess.Message = [preprocess.Message msg];     
elseif (preprocess.Evaluation == 2)
    msg = sprintf(' Testing on File %s, ', preprocess.test_file);
    preprocess.Message = [preprocess.Message msg];     
elseif (preprocess.Evaluation == 3)
    msg = sprintf(' Training on File %s, ', preprocess.input_file);
    preprocess.Message = [preprocess.Message msg];     
elseif (preprocess.Evaluation == 4)
    msg = sprintf(' Testing on File %s, ', preprocess.input_file);
    preprocess.Message = [preprocess.Message msg];     
else 
    msg = sprintf(' Train-Test Split, Boundary: %d, ', preprocess.TrainTestSplitBoundary);
    preprocess.Message = [preprocess.Message msg]; 
end;

% if (preprocess.MultiClassType == 0)
%     msg = sprintf(' Classification, ', preprocess.TrainTestSplitBoundary);
%     preprocess.Message = [preprocess.Message msg]; 
% elseif (preprocess.MultiClassType == 1) 
%     msg = sprintf(' Multiclass Classification Wrapper, ', preprocess.NumCrossFolder);
%     preprocess.Message = [preprocess.Message msg];     
% elseif (preprocess.MultiClassType == 2) 
%     msg = sprintf(' Multilabel Classification Wrapper, ', preprocess.NumCrossFolder);
%     preprocess.Message = [preprocess.Message msg];     
% elseif (preprocess.MultiClassType == 3) 
%     msg = sprintf(' Multiclass Active Learning Wrapper, ', preprocess.NumCrossFolder);
%     preprocess.Message = [preprocess.Message msg];     
% end;

% if (preprocess.SVD == 1)
%     msg = sprintf(' SVD ');
%     preprocess.Message = [preprocess.Message msg]; 
% end;

if (preprocess.Shuffled == 1)
    msg = sprintf(' Shuffled ');
    preprocess.Message = [preprocess.Message msg]; 
end;

% if (preprocess.Sparse == 1)
%     msg = sprintf(' Sparse ');
%     preprocess.Message = [preprocess.Message msg]; 
% end;

% if (preprocess.MultiClass.CodeType == 0) 
%     msg = sprintf(' Coding: 1-vs-r ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.CodeType == 1)
%     msg = sprintf(' Coding: 1-vs-1 ');
%     preprocess.Message = [preprocess.Message msg]; 
% elseif (preprocess.MultiClass.CodeType == 2)
%     msg = sprintf(' Coding: ECOC15_5 ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.CodeType == 3)
%     msg = sprintf(' Coding: ECOC63_31 ');
%     preprocess.Message = [preprocess.Message msg]; 
% elseif (preprocess.MultiClass.CodeType == 4)
%     msg = sprintf(' Coding: Random ');
%     preprocess.Message = [preprocess.Message msg]; 
% end;    

% if (preprocess.MultiClass.LossFuncType == 0) 
%     msg = sprintf(' Loss: L1 ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.LossFuncType == 1)
%     msg = sprintf(' Loss: Exp ');
%     preprocess.Message = [preprocess.Message msg]; 
% elseif (preprocess.MultiClass.LossFuncType == 2)
%     msg = sprintf(' Loss: (1-Y)+ ');
%     preprocess.Message = [preprocess.Message msg];
% end;    
% 
% if (preprocess.MultiClass.UncertaintyFuncType == 0) 
%     msg = sprintf(' Uncertainty: L1 ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.UncertaintyFuncType == 1)
%     msg = sprintf(' Uncertainty: Exp ');
%     preprocess.Message = [preprocess.Message msg]; 
% elseif (preprocess.MultiClass.UncertaintyFuncType == 2)
%     msg = sprintf(' Uncertainty: (1-Y)+ ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.UncertaintyFuncType == 3)
%     msg = sprintf(' Uncertainty: Min Margin ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.UncertaintyFuncType == 4)
%     msg = sprintf(' Uncertainty: -ln(1+x) ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.UncertaintyFuncType == 5)
%     msg = sprintf(' Uncertainty: Random ');
%     preprocess.Message = [preprocess.Message msg];
% end;    
% 
% if (preprocess.MultiClass.ProbEstimation == 0) 
%     msg = sprintf(' Best Worse ');
%     preprocess.Message = [preprocess.Message msg];
% elseif (preprocess.MultiClass.ProbEstimation == 1)
%     msg = sprintf(' Uniform Guess ');
%     preprocess.Message = [preprocess.Message msg]; 
% elseif (preprocess.MultiClass.ProbEstimation == 2)
%     msg = sprintf(' Binary Class ');
%     preprocess.Message = [preprocess.Message msg];     
% end;

% if (preprocess.DataSampling == 1)
%     msg = sprintf(' Sampling Rate: %d', preprocess.DataSamplingRate);
%     preprocess.Message = [preprocess.Message msg];    
% end;

% num_folder = preprocess.NumFolder;
% num_cross_folder = preprocess.NumCrossFolder;; 

% load in the data
if ((~isfield(preprocess, 'input_file')) || (isempty(preprocess.input_file))), 
    error('The input file is not provided!');
end;

input_file = preprocess.input_file;
output_file = preprocess.output_file;
pred_file = preprocess.pred_file;

fid = fopen(output_file, preprocess.OutputFlag);
if (fid < 0),
    if ((~isfield(preprocess, 'test_file')) || (isempty(preprocess.test_file))), 
        output_file = sprintf('%s.result', input_file);
    else
        output_file = sprintf('%s.result', preprocess.test_file);
    end;
    preprocess.output_file = output_file;
    fid = fopen(output_file, preprocess.OutputFlag);
end;
if (preprocess.OutputFormat == 0),
    fprintf(fid, '\nProcessing Filename: %s\n', preprocess.input_file); 
    fprintf(fid, 'Classifier:%s\nMessage:%s\n',classifier, preprocess.Message);
end;
fclose(fid);    

fid = fopen(pred_file, preprocess.OutputFlag);
if (fid < 0),
    if ((~isfield(preprocess, 'test_file')) || (isempty(preprocess.test_file))), 
        pred_file = sprintf('%s.pred', input_file);
    else
        pred_file = sprintf('%s.pred', preprocess.test_file);
    end;
    preprocess.pred_file = pred_file;
    fid = fopen(pred_file, preprocess.OutputFlag);
end;
if (preprocess.PredFormat == 0),
    fprintf(fid, '\nProcessing Filename: %s\n', preprocess.input_file); 
    fprintf(fid, 'Classifier:%s\nMessage:%s\n',classifier, preprocess.Message);
end;
fclose(fid);    

% in MIL, we use a function to read the input data and convert into bag
% formats; sparse format is not currently supported 

% if (preprocess.InputFormat == 1),
%     strcmd = sprintf('!perl %s/ConvertFileInput.pl %s %s.stdout', root, input_file, input_file);
%     fprintf('!perl %s/ConvertFileInput.pl %s %s.stdout\n', root, input_file, input_file); 
%     eval(strcmd);
%     D = dlmread(sprintf('%s.stdout', input_file));
% else
%     D = dlmread(input_file);
% end;
% 
% if (preprocess.Sparse == 1),
%     D = spconvert(D);
% end;

% Automatically judge whether the shot information is available
% if (preprocess.ShotAvailable < 0),
%    if (length(unique(D(:, size(D, 2)))) > 10), 
%        preprocess.ShotAvailable = 1;
%    else
%        preprocess.ShotAvailable = 0;
%    end;
%    fprintf('Automatically detect preprocess.ShotAvailable to be %d \n', preprocess.ShotAvailable );
% end;

fprintf('Finished loading %s.............\n', input_file);
fprintf('Output Results to %s.............\n', output_file);
fprintf('Output Predictions to %s.............\n', pred_file);

switch (preprocess.Evaluation)
    case 0
        % Train Test
        EvaluationHandle = @MIL_Train_Test_Validate;
    case 1
        % Cross Validate
        EvaluationHandle = @MIL_Cross_Validate;
    case 2
        % Test File Validate
        if ((~isfield(preprocess, 'test_file')) | (isempty(preprocess.test_file))), 
            error('The test file is not provided!');
        end;
       %D_test = dlmread(preprocess.test_file, ',');
       %fprintf('Finished loading the test file %s.............\n', preprocess.test_file);
       %preprocess.TrainTestSplitBoundary = size(D, 1);
        preprocess.Shuffled = 0;
        %D = [D; D_test];
        EvaluationHandle = @MIL_Train_Test_Validate;   
    case 3
        % Training Only
        EvaluationHandle = @MIL_Train_Validate;
    case 4
        % Testing Only
        EvaluationHandle = @MIL_Test_Validate;
    case 5
        EvaluationHandle = @MIL_Leave_One_Out;           
        preprocess.Shuffled = 0;
end;

fhandle = @MIL_Train_Test_Simple;   %currently only address the binary MIL classification probleem 
% switch (preprocess.MultiClassType)
%     case 0
%         fhandle = @Train_Test_Simple;
%     case 1
%         fhandle = @Train_Test_Multiple_Class;
%     case 2
%         fhandle = @Train_Test_Multiple_Label;
%     case 3
%         fhandle = @Train_Test_Multiple_Class_AL;
% end;
       
fprintf('Classifier:%s\nMessage:%s\n',classifier, preprocess.Message);
if ((preprocess.Evaluation >= 3) && (preprocess.Evaluation <= 4))
    run = feval(EvaluationHandle, input_file, classifier);
else
    run = feval(EvaluationHandle, input_file, fhandle, classifier);
end;

OutputResult = [];
if (isfield(run, 'BagAccu')), OutputResult = [OutputResult sprintf('Bag label accuracy = %f, ', run.BagAccu)]; end;
if (isfield(run, 'InstAccu')), OutputResult = [OutputResult sprintf('Instance label accuracy = %f, ', run.InstAccu)]; end;

%if (isfield(run, 'Err')), OutputResult = [OutputResult sprintf('Error = %f, ', run.Err)]; end;
%if (isfield(run, 'Prec')), OutputResult = [OutputResult sprintf('Precision = %f, ', run.Prec)]; end;
%if (isfield(run, 'Rec')), OutputResult = [OutputResult sprintf('Recall = %f, ', run.Rec)]; end;
%if (isfield(run, 'F1')), OutputResult = [OutputResult sprintf('F1 = %f, ', run.F1)]; end;
%if (isfield(run, 'Micro_Prec')), OutputResult = [OutputResult sprintf('Micro_Precision = %f, ', run.Micro_Prec)]; end;
%if (isfield(run, 'Micro_Rec')), OutputResult = [OutputResult sprintf('Micro_Recall = %f, ', run.Micro_Rec)]; end;
%if (isfield(run, 'Micro_F1')), OutputResult = [OutputResult sprintf('Micro_F1 = %f, ', run.Micro_F1)]; end;
%if (isfield(run, 'Macro_Prec')), OutputResult = [OutputResult sprintf('Macro_Precision = %f, ', run.Macro_Prec)]; end;
%if (isfield(run, 'Macro_Rec')), OutputResult = [OutputResult sprintf('Macro_Recall = %f, ', run.Macro_Rec)]; end;
%if (isfield(run, 'Macro_F1')), OutputResult = [OutputResult sprintf('Macro_F1 = %f, ', run.Macro_F1)]; end;
%if (isfield(run, 'AvgPrec')), OutputResult = [OutputResult sprintf('MAP = %f, ', run.AvgPrec)]; end;
%if (isfield(run, 'BaseAvgPrec')), OutputResult = [OutputResult sprintf('MBAP = %f, ', run.BaseAvgPrec)]; end;

fprintf('%s\n', OutputResult);
if (~isempty(output_file)) 
    if ((preprocess.OutputFormat == 0) | (preprocess.OutputFormat == 1)),
        fid = fopen(output_file, 'a');
        fprintf(fid, '%s\n', OutputResult);
        fclose(fid);    
    end;
end;
if (~isempty(pred_file))
    fid = fopen(pred_file, 'w');
    fprintf(fid, 'Testing Bag Label Evaluation:\n');
    fprintf(fid, 'Index\tProbability\tPredict\tTruth\n');
    fprintf(fid, '%d\t%g\t%d\t%d\n', run.bag_pred');
    
%     if (preprocess.OutputFormat == 0),
%         if (size(run.Y_pred, 2) == 4),
%             fprintf(fid, 'Index\tProb\t\tPred\tTruth\n');
%             fprintf(fid, '%d\t%f\t%d\t%d\n', run.Y_pred');
%         else
%             fprintf(fid, 'Index\tProb\t\tPred\tTruth\tShotID\n');
%             fprintf(fid, '%d\t%f\t%d\t%d\t%d\n', run.Y_pred');
%         end;
%     elseif (preprocess.OutputFormat == 1), % Only work when number of classes is 2
%         fprintf(fid, 'Label: '); fprintf(fid, '%d ', preprocess.OrgClassSet); fprintf(fid, '\n');
%         ProbClass1 = run.Y_pred(:, 2) .* (run.Y_pred(:, 3) == 1)  + (1 - run.Y_pred(:, 2)) .* (run.Y_pred(:, 3) ~= 1);
%         fprintf(fid, '%d\t%f\t%f\n', [preprocess.OrgClassSet(run.Y_pred(:, 3)) ProbClass1 (1 - ProbClass1)]');
%     end;
    fclose(fid);
end;

function Report_Error()
fprintf(' Example: Suppose you are in root directory\n');
fprintf(' Example: test_classify.exe \"classify -t demo/DataExample1.train.txt -sh 1 -- train_only -m demo/DataExample1.libSVM.model -- LibSVM -Kernel 0 -CostFactor 3\" \n');
fprintf(' Example: test_classify.exe \"classify -t demo/DataExample1.test.txt -sh 1 -- test_only -m demo/DataExample1.libSVM.model -- LibSVM -Kernel 0 -CostFactor 3\" \n');
fprintf(' Please refer to http://finalfantasyxi.inf.cs.cmu.edu/tmp/MATLABArsenal.htm for more examples \n');
error('  The command must begin with Classify     ');
