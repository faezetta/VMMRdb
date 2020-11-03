function  [test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = MIL_Classify(classifier_str, train_bags, test_bags)

global preprocess; 

% if ((preprocess.FLD > 0) & (size(X_train, 2) > preprocess.FLD)) % FLD reduction
%     fprintf('FLD: reduce to %d dimension\n', preprocess.FLD);
%     % [X_train, Y_train] = FLD(X_train, Y_train, X_test, Y_test, preprocess.FLD);
%     [X_train, discrim_vec] = LDA(X_train, Y_train, preprocess.FLD)
%     X_test = X_test * discrim_vec(:,1:preprocess.FLD);
%     if (~isempty(preprocess.DimReductionFile)),
%         dlmwrite(preprocess.DimReductionFile, [X_train Y_train; X_test Y_test]);    
%     end;
% end;

[classifier, para, additional_classifier] = ParseCmd(classifier_str, '--');

[test_bag_label, test_inst_label, test_bag_prob, test_inst_prob] = feval(classifier, para, train_bags, test_bags); 

% if (strcmpi(classifier, 'ZeroR')~=0),
%     Y_compute = zeros(size(Y_test)); 
%     Y_prob = zeros(size(Y_test));       
% elseif (isempty(additional_classifier)), 
%     [Y_compute, Y_prob] = feval(classifier, para, X_train, Y_train, X_test, Y_test, num_class); 
% else
%     [Y_compute, Y_prob] = feval(classifier, additional_classifier, para, X_train, Y_train, X_test, Y_test, num_class); 
% end;

%all MIL methods provide bag labels, some also provide instance labels
if (isempty(test_bag_label)),
    error('testing bag labels are empty');      
end;
