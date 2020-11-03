function  [Y_compute, Y_prob] = Classify(classifier_str, X_train, Y_train, X_test, Y_test, num_class)

global preprocess; 
if ((preprocess.FLD > 0) & (size(X_train, 2) > preprocess.FLD)) % FLD reduction
    fprintf('FLD: reduce to %d dimension\n', preprocess.FLD);
    % [X_train, Y_train] = FLD(X_train, Y_train, X_test, Y_test, preprocess.FLD);
    [X_train, discrim_vec] = LDA(X_train, Y_train, preprocess.FLD)
    X_test = X_test * discrim_vec(:,1:preprocess.FLD);
    if (~isempty(preprocess.DimReductionFile)),
        dlmwrite(preprocess.DimReductionFile, [X_train Y_train; X_test Y_test]);    
    end;
end;

[classifier, para, additional_classifier] = ParseCmd(classifier_str, '--');

if (strcmpi(classifier, 'ZeroR')~=0),
    Y_compute = zeros(size(Y_test)); 
    Y_prob = zeros(size(Y_test));       
elseif (isempty(additional_classifier)), 
    [Y_compute, Y_prob] = feval(classifier, para, X_train, Y_train, X_test, Y_test, num_class); 
else
    [Y_compute, Y_prob] = feval(classifier, additional_classifier, para, X_train, Y_train, X_test, Y_test, num_class); 
end;

if (isempty(Y_compute) ),
    error('Y_compute or Y_prob is empty!');
end;

% if (preprocess.Ensemble == 1) %Up-Sampling
%     Y_compute = UpSampling(classifier, para, X_train, Y_train, X_test, Y_test, num_class);           
% elseif (preprocess.Ensemble == 2) %Down-Sampling
%     Y_compute = DownSampling(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (preprocess.Ensemble == 3) %Meta Classify, Majority Vote
%     Y_compute = MetaClassifyWithVoting(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (preprocess.Ensemble == 4) %Hierachy Classify
%     Y_compute = HierarchyClassify_Dev(classifier, para, X_train, Y_train, X_test, Y_test, num_class);   
% elseif (preprocess.Ensemble == 5) %Sum Rule
%     Y_compute = MetaClassifyWithSumRule(classifier, para, X_train, Y_train, X_test, Y_test, num_class);   
% elseif (preprocess.Ensemble == 6) %Stacking Classification
%     Y_compute = StackedClassify(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (preprocess.Ensemble == 7) %Active Learning
%     [Y_compute, Y_prob] = ActiveLearning(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (preprocess.Ensemble == 8) %Meta Classify with multiple feature sets
%     [Y_compute, Y_prob] = MetaClassifyWithMultiFSet(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (preprocess.Ensemble == 9) %Learn Order
%     [Y_compute, Y_prob] = LearnOrder_PairWise(classifier, para, X_train, Y_train, X_test, Y_test, num_class);    
% else
%     [Y_compute, Y_prob] = Classify(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% end;

% if (strcmp(classifier, 'SMO')~=0)
%     [Y_compute, Y_prob] = WekaClassify('SMO', para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (strcmp(classifier, 'SVM_LIGHT')~=0)
%     [Y_compute, Y_prob] = svm_light(para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (strcmp(classifier, 'SVM_LIGHT_TRAN')~=0)
%     [Y_compute, Y_prob] = svm_light_transductive(para, X_train, Y_train, X_test, Y_test, num_class);    
% elseif (strcmp(classifier, 'MY_SVM')~=0)
%     [Y_compute, Y_prob] = mySVM(para, X_train, Y_train, X_test, Y_test, num_class);    
% elseif (strcmp(classifier, 'j48')~=0)
%     [Y_compute, Y_prob] = WekaClassify('j48.J48', para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (strcmp(classifier, 'NaiveBayes')~=0)
%     [Y_compute, Y_prob] = WekaClassify(classifier, para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (strcmp(classifier, 'LinearRegression')~=0)
%     [Y_compute, Y_prob] = WekaClassify(classifier, para, X_train, Y_train, X_test, Y_test, num_class);    
% elseif (strcmp(classifier, 'LWR')~=0)
%     [Y_compute, Y_prob] = WekaClassify(classifier, para, X_train, Y_train, X_test, Y_test, num_class);    
% elseif (strcmp(classifier, 'kNN')~=0)
%     [Y_compute, Y_prob] = kNN(para, X_train, Y_train, X_test, Y_test, num_class);
% elseif (strcmp(classifier, 'ME')~=0)
%     [Y_compute, Y_prob] = IIS_classify(X_train, Y_train, X_test);
%     %Y_compute = train_cg_multiple(X_train, Y_train, X_test);
% elseif (strcmp(classifier, 'NeuralNet')~=0)
%     [Y_compute, Y_prob] = NeuralNet(X_train, Y_train, X_test, para); 
% elseif (strcmp(classifier, 'NNSearch')~=0)    
%     [Y_compute, Y_prob] = NNSearch(para, X_train, X_test); 
% elseif (strcmp(classifier, 'LogitReg')~=0)
%     [Y_compute, Y_prob] = LogitReg(X_train, Y_train, X_test);   
% elseif (strcmp(classifier, 'LogitRegKernel')~=0)
%     [Y_compute, Y_prob] = LogitRegKernel(X_train, Y_train, X_test, para);       
% end;