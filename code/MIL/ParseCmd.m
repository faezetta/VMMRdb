function [classifier, para, other_classifier] = ParseCmd(classifier_str, delimiter)

% Extract the parameters and classifiers
[classifier rem] = strtok(classifier_str);

para = [];
additional_classifier = [];
while (~isempty(rem)),
    [cell_para rem] = strtok(rem);
    if (strcmp(cell_para, delimiter)), break; end;
    para = [para cell_para ' '];
end;

% remove the leading blanks
[r,c] = find((rem ~=0) & ~isspace(rem));
other_classifier = rem(min(c):length(rem));
