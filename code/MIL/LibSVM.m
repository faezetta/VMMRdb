function  [Y_compute, Y_prob] = libSVM(para, X_train, Y_train, X_test, Y_test)
   
global temp_train_file temp_test_file temp_output_file temp_model_file libSVM_dir; 

num_class = 2;
p = str2num(char(ParseParameter(para, {'-Kernel';'-KernelParam'; '-CostFactor'; '-NegativeWeight'; '-Threshold'}, {'2';'0.05';'1';'1';'0'})));

switch p(1)
    case 0
      s = '';      
    case 1
      s = sprintf('-d %.10g -g 1', p(2));
    case 2
      s = sprintf('-g %.10g', p(2));
    case 3
      s = sprintf('-r %.10g', p(2)); 
    case 4
      s = sprintf('-u "%s"', p(2));
end
        
% set up the commands
train_cmd = sprintf('!svm\\svmtrain -b 1 -s 0 -t %d %s -c %f -w1 1 -w0 %f %s %s > log1', p(1), s, p(3), p(4), temp_train_file, temp_model_file);
test_cmd = sprintf('!svm\\svmpredict -b 1 %s %s %s > log1', temp_test_file, temp_model_file, temp_output_file);

[num_train_data, num_feature] = size(X_train);   
[num_test_data, num_feature] = size(X_test);   

if (~isempty(X_train)),
    fid = fopen(temp_train_file, 'w');
    for i = 1:num_train_data,
        Xi = X_train(i, :);
        % Write label as the first entry
        s = sprintf('%d ', Y_train(i));
        % Then follow 'feature:value' pairs
        ind = find(Xi);
        s = [s sprintf(['%i:' '%.10g' ' '], [ind' full(Xi(ind))']')];
        fprintf(fid, '%s\n', s);
    end
    fclose(fid);
    % train the svm
    fprintf('Running: %s..................\n', train_cmd);
    eval(train_cmd);
end;

% Prediction
fid = fopen(temp_test_file, 'w');
for i = 1:num_test_data,
  Xi = X_test(i, :);
  % Write label as the first entry
  s = sprintf('%d ', Y_test(i));
  % Then follow 'feature:value' pairs
  ind = find(Xi);
  s = [s sprintf(['%i:' '%.10g' ' '], [ind' full(Xi(ind))']')];
  fprintf(fid, '%s\n', s);
end
fclose(fid);
fprintf('Running: %s..................\n', test_cmd);
eval(test_cmd);

fid = fopen(temp_output_file, 'r');
line = fgets(fid);

Y = fscanf(fid, '%f');
fclose(fid);

Y = (reshape(Y, num_class + 1, num_test_data))';
Y_compute = int16(Y(:, 1));

if isempty(strfind(line, 'labels 1 0'))
    Y_prob = Y(:, 3);
else
    Y_prob = Y(:, 2);
end



