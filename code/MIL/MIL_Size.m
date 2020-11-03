function [num_bag, num_inst, num_feature] = MIL_Size(bags);

num_bag = length(bags);

if num_bag == 0
    num_inst = 0;
    num_feature = -1;
else
    num_inst = 0;
    for i = 1 : num_bag, num_inst = num_inst + size(bags(i).instance, 1); end;
    num_feature = size(bags(1).instance, 2);
end
    
        