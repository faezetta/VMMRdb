function [bags, num_data, num_feature] = MIL_Data_Load(filename)
    global preprocess;

    matrix_file = [filename '.matrix'];
    label_file  = [filename '.label'];
    insts = [];

    if preprocess.InputFormat == 1
        %sparse input format
        strcmd = sprintf('!ReadInput.pl %s 1', filename);
        eval(strcmd);
        D = load(matrix_file);
        insts = spconvert(D);
    else
        strcmd = sprintf('!ReadInput.pl %s', filename);
        % If any problem:::::::::::::::::::::::::::::::::::::::::::::::::::::::
        % run matlab from shell
        % dos2unix ReadInput.pl ReadInput.pl %%% Use the non converted file
        % #!/usr/bin/env perl   %% adding shebang at the beginning of the file
        % :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        eval(strcmd);
        insts = load(matrix_file);
    %     instsTmp = importdata(matrix_file);
    %     insts = instsTmp.data;
    end
            
    fid = fopen(label_file, 'r');
    if fid == -1, error('The label file is not generated, quitting...'); end;

    nbag = 0;
    prev_bag_name = '';
    ninst = 0;
    idx = 0;
    istest = ~isempty(findstr(filename, 'test'));
    while feof(fid) == 0

        line = strtrimm(fgets(fid));
        elems = strsplit(' ',line);    %instance_name, bag_name, label

        bag_name = cell2mat(elems(2));
        if strcmp(bag_name, prev_bag_name) == 0     %change of bag
            if (nbag >= 1)
                instTemp = insts(idx + 1:idx + ninst, :);
    %             bags(nbag).instance = instTemp;
                bags(nbag).instance = instTemp(~any(isnan(instTemp),2),:);
                bags(nbag).inst_name(any(isnan(instTemp),2)) = [];
                bags(nbag).inst_label(any(isnan(instTemp),2)) = [];
                bags(nbag).label = any(bags(nbag).inst_label);
                if istest  % Keeping track of deleted nan rows
                    bagstmp(nbag).nanInd = any(isnan(instTemp),2);
                end
                idx = idx + ninst;
            end;
            nbag = nbag + 1;
            bags(nbag).name = bag_name;
            prev_bag_name = bag_name;
            ninst = 0;
        end

        ninst = ninst + 1;
        bags(nbag).inst_name(ninst) = elems(1);
        label = cell2mat(elems(3));
        bags(nbag).inst_label(ninst) = strcmp(label,'1');   %the positive label must be set to 1
    end;

    if (nbag >= 1)
        instTemp = insts(idx + 1:idx + ninst, :);
    %     bags(nbag).instance = instTemp;
        bags(nbag).instance = instTemp(~any(isnan(instTemp),2),:); % insts(idx + 1:idx + ninst,:);
        bags(nbag).inst_name(any(isnan(instTemp),2)) = [];
        bags(nbag).inst_label(any(isnan(instTemp),2)) = [];
        bags(nbag).label = any(bags(nbag).inst_label);
        if istest
            bagstmp(nbag).nanInd = any(isnan(instTemp),2);
        end
    end;
    fclose(fid);

    num_data = length(bags);
    num_feature = size(bags(1).instance, 2);

    % normalize the data set
    if (preprocess.Normalization == 1)
        bags = MIL_Scale(bags);
    end;

    % randomize the data
    rand('state',sum(100*clock));
    if (preprocess.Shuffled == 1) %Shuffle the datasets
        Vec_rand = rand(num_data, 1);
        [B, Index] = sort(Vec_rand);
        bags = bags(Index);
    end;

    if istest
        [pathstr,name,~] = fileparts(filename);
        nanInd = vertcat(bagstmp.nanInd);
        save([pathstr filesep name '_nans.mat'],'nanInd');
    end


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

function parts = strsplit(splitstr, str, option)
    %STRSPLIT Split string into pieces.
    %
    %   STRSPLIT(SPLITSTR, STR, OPTION) splits the string STR at every occurrence
    %   of SPLITSTR and returns the result as a cell array of strings.  By default,
    %   SPLITSTR is not included in the output.
    %
    %   STRSPLIT(SPLITSTR, STR, OPTION) can be used to control how SPLITSTR is
    %   included in the output.  If OPTION is 'include', SPLITSTR will be included
    %   as a separate string.  If OPTION is 'append', SPLITSTR will be appended to
    %   each output string, as if the input string was split at the position right
    %   after the occurrence SPLITSTR.  If OPTION is 'omit', SPLITSTR will not be
    %   included in the output.

    %   Author:      Peter J. Acklam
    %   Time-stamp:  2004-09-22 08:48:01 +0200
    %   E-mail:      pjacklam@online.no
    %   URL:         http://home.online.no/~pjacklam

   nargsin = nargin;
   error(nargchk(2, 3, nargsin));
   if nargsin < 3
      option = 'omit';
   else
      option = lower(option);
   end

   splitlen = length(splitstr);
   parts = {};

   while 1

      k = strfind(str, splitstr);
      if isempty(k)
         parts{end+1} = str;
         break
      end

      switch option
         case 'include'
            parts(end+1:end+2) = {str(1:k(1)-1), splitstr};
         case 'append'
            parts{end+1} = str(1 : k(1)+splitlen-1);
         case 'omit'
            parts{end+1} = str(1 : k(1)-1);
         otherwise
            error(['Invalid option string -- ', option]);
      end


      str = str(k(1)+splitlen : end);

   end
