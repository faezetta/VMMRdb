function label = prob2label(prob, bags)

num_bag = length(bags);

idx = 0;
for i=1:num_bag
    num_inst = size(bags(i).instance, 1);    
    [sort_ret, sort_idx] = sort(prob(idx+1 : idx+num_inst));        
    
    label(idx+1 : idx+num_inst) = 0;
    label(idx + sort_idx(num_inst)) = 1; 
    
    idx = idx + num_inst;
end
