function MIL_Data_Save(filename, bags)

fid = fopen(filename, 'w');
if fid == -1, error('Cannot open the file for saving'); end;

for i = 1:length(bags)
    [num_inst, num_feature] = size(bags(i).instance);
    for j = 1:num_inst
        str = [bags(i).name ',' cell2mat(bags(i).inst_name(j)) ',' feature_line(bags(i).instance(j,:)) num2str(bags(i).inst_label(j)) '\n'];
        fprintf(fid, str);
    end
end
fclose(fid);

function str = feature_line(inst)

str = '';
for i = 1:length(inst)    
    str = [str num2str(inst(i)) ','];
end