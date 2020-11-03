function [acc, ncorrect] = evaluate_by_inst_choice(bags, labels)
nbag = length(bags);

ninst = 0;
for i=1:nbag, ninst = ninst + size(bags(i).instance, 1); end;
if (ninst ~= length(labels)) || (ninst == 0)
    acc = -1;
    return;     %return -1 if the number of bags dismatch the number of labels
end

ncorrect = 0;
npositive = 0;
idx = 1;
for i=1:nbag
    if bags(i).label == 0
        idx = idx + size(bags(i).instance, 1);
    else
        correct = 0;
        for j=1:length(bags(i).inst_label)
            if (bags(i).inst_label(j) ==1) && (labels(idx) == 1)
                correct = 1;
            end
            idx = idx + 1;
        end
        npositive = npositive + 1;
        if correct == 1, ncorrect = ncorrect + 1; end;
    end
end
acc = ncorrect / npositive;
