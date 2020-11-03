function OutputRet(filename, bags, threshold)

fid = fopen(filename, 'w');

for i=1:length(bags)
    scores = bags(i).inst_prob;
    for j=1:length(scores)
        if scores(j) > threshold
            fprintf(fid, '1 %12.8f\n', scores(j));
        else
            fprintf(fid, '-1 %12.8f\n', scores(j));
        end
    end    
end
fclose(fid);