function [class_set, num_class] = GetClassSet(Y)

global preprocess;
class_set = unique(Y);
if (isfield(preprocess, 'ClassSet')),
    class_set = preprocess.ClassSet;
end;
num_class = length(class_set);