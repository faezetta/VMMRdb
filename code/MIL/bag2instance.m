function [inst, inst_label] = bag2instance(bags)

num_bag = length(bags);

if num_bag == 0
    inst = [];
    inst_label = [];
end

idx = 0;
for i=1:num_bag
    num_inst = size(bags(i).instance, 1);
    inst(idx+1 : idx+num_inst, :) = bags(i).instance;    
    inst_label(idx+1 : idx+num_inst) = bags(i).inst_label;    
    idx = idx + num_inst;
end
inst_label = inst_label';