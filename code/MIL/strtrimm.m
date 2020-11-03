function s = strtrimm(str)

%trim the linebreak and space on both sides of the string

s = str;

i = 1;
while isspace(str(i)) && i <= length(str)
    i = i+1;    
end

j = length(str);
while isspace(str(j)) && j >= 1
    j = j - 1;    
end

if (i <= j)
    s = str(i:j);
else
    s = '';
end

