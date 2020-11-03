function mil2sil(input_file, output_file, is_text)

global preprocess;
preprocess.Normalization = 0;
preprocess.Shuffled = 0;
    
if is_text == 0    
    preprocess.InputFormat = 0;   
    [bags, num_bag, num_feature] = MIL_Data_Load(input_file);
    
    label = zeros(num_bag, 1); 
    for i = 1 : num_bag
        insts(i,:) = mean(bags(i).instance, 1);
        label(i) = bags(i).label;
    end 
else    
    preprocess.InputFormat = 1;    
    [bags, num_bag, num_feature] = MIL_Data_Load(input_file);
    
    label = zeros(num_bag, 1);
    for i = 1 : num_bag
        num_inst = size(bags(i).instance, 1);
        for j = 1 : num_inst
            tmp_inst = nonzeros(bags(i).instance(j,:));
            min_elem = min(tmp_inst);
            bags(i).instance(j,:) = round(bags(i).instance(j,:) ./ min_elem);
        end

        insts(i,:) = sum(bags(i).instance, 1);
        norm = sum(insts(i,:));
        insts(i,:) = insts(i,:) / norm;
        label(i) = bags(i).label;
    end    
end

fid = fopen(output_file, 'w');
for i = 1:num_bag,
    Xi = insts(i, :);
    % Write label as the first entry
    s = sprintf('%d ', label(i));
    % Then follow 'feature:value' pairs
    ind = find(Xi);
    s = [s sprintf(['%i:' '%.10g' ' '], [ind' full(Xi(ind))']')];
    fprintf(fid, '%s\n', s);
end
fclose(fid);

