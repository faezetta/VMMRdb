function [acc, ncorrect] = MIL_Inst_Evaluate(bags, labels)
nbag = length(bags);
ninst = 0;
for i=1:nbag, ninst = ninst + size(bags(i).instance, 1); end;

if (ninst ~= length(labels)) || (ninst == 0)
    acc = -1;
    return;     %return -1 if the number of bags dismatch the number of labels
end

ncorrect = 0;
idx = 1;
for i=1:nbag
    for j=1:length(bags(i).inst_label)
        if bags(i).inst_label(j) == labels(idx)
            ncorrect = ncorrect + 1;
        end
        idx = idx + 1;
    end
end
acc = ncorrect / ninst;
