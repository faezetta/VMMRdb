function [acc, ncorrect] = MIL_Bag_Evaluate(bags, labels)
nbag = length(bags);

if (nbag ~= length(labels)) || (nbag == 0)
    acc = -1;
    return;     %return -1 if the number of bags dismatch the number of labels
end

ncorrect = 0;
for i=1:nbag
    if bags(i).label == labels(i)
        ncorrect = ncorrect + 1;
    end
end
acc = ncorrect / nbag;
