function p = ParseParameter(str, format, initpara, display)

if (nargin < 4), display = 0; end;

p = initpara;
rem = str;

if (display > 0), fprintf('Paras: '); end;
while (~isempty(rem)), 
    [tok, rem] = strtok(rem);
    for i = 1:length(format)
        if strcmpi(char(format{i}), tok)
            break;
        end;
    end;
    if (strcmpi(char(format{i}), tok)), 
        [tok, rem1] = strtok(rem);
        if ((isempty(tok)) | (tok(1) == '-')), % on/off parameters
            p{i} = '1';
            if (display > 0), fprintf('%s: %s, ', char(format{i}), p{i}); end;
        else % real/binary value parameters
            p{i} = tok; rem = rem1;
            if (display > 0), fprintf('%s: %s, ', char(format{i}), p{i}); end;
        end;
    end;
end;
if (display > 0), fprintf('\n'); end;
