function bags = MIL_Scale(bags, lb, ub)

if nargin < 3, ub = 1; end;
if nargin < 2, lb = 0; end;

for i=1:length(bags)
    fea_max(i,:) = max(bags(i).instance,[],1);
    fea_min(i,:) = min(bags(i).instance,[],1);
end

fea_max = max(fea_max,[],1);
fea_max = max([fea_max; ones(1, length(fea_max))]);

fea_min = min(fea_min,[],1);
fea_min = min([fea_min; zeros(1, length(fea_min))]);

for i=1:length(bags)    
    [ninst, nfea] = size(bags(i).instance);

    max_mat = repmat(fea_max, ninst, 1); 
    min_mat = repmat(fea_min, ninst, 1);
    
    bags(i).instance = lb + ((bags(i).instance - min_mat) ./ (max_mat - min_mat)) .* (ub-lb);    
end

