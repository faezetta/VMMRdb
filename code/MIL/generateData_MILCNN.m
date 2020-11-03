%% Generate bags using saliency detection for the MIL-CNN training
% SAve results in form of bags and csv
function generateData_MILCNN()   
    regionType = 'saliency';
    regionNum = 14;     % number of salient/grid regions
    imSize = [224 224];
    edgeFilter = 0;
    
    % Get images 
    mergedDir = 'D:\Dropbox\Deep Learning\CompCars_VMMR\_Training\Merged'; 
    destDir = 'D:\Dropbox\Deep Learning\CompCars_VMMR\_Training\Merged_MILCNN'; 
    folders = {'val','train'};
    exts = {'jpg','png'};
    
    traintestData = []; 
    for jj = 1:length(folders)
        dbfiles = dir([mergedDir filesep folders{jj}]);
        dbClasses = {dbfiles([dbfiles.isdir]).name};   
        dbfiles(~[dbfiles.isdir]) = []; 
        for kk = 3 : length(dbfiles)
            f = cellfun( @(x) dir( fullfile( mergedDir, folders{jj}, dbfiles(kk).name, ['*.',x] ) ), exts, 'Un', 0 ); 
            files = vertcat(f{:});
            for ii=1:length(files)
                cur_filename = files(ii).name;
                disp(['Extracting instances for: {image:' cur_filename '- ' num2str(ii) '/' num2str(length(files)) ' }']);

                curDir = [destDir filesep folders{jj} filesep dbfiles(kk).name filesep cur_filename(1:end-4)];
                if ~exist(curDir, 'dir') mkdir(curDir);  end
                
                curImg = imread([mergedDir filesep folders{jj} filesep dbfiles(kk).name filesep cur_filename]);
                if strcmp(regionType,'saliency')
                    curRegions = saliencyBMS_MILCNN(curImg, cur_filename, curDir, regionNum);
                elseif strcmp(regionType,'grid')
                    curRegions = imageGrid(curImg, imSize, regionNum);
                else
                    curRegions = extractRegions(curImg, cur_filename, [expDir filesep imageDir], imSize);
                end
%                 if ((mod(i,100)==0) || (i==maxInd))
%                     save(['data-' num2str(regionNum) filesep 'fullData-full-' type '-' num2str(i) '.mat'],'traintestData');
%                     traintestData = [];
%                 end
            end
            display([folders{jj} ':  class ' num2str(kk) '/' num2str(length(dbfiles)) ' -- ' dbfiles(kk).name]);
        end
        clear files f dbClasses dbfiles
    end
end

%% Manual region extraction
function [curRegions] = extractRegions(img, fileName, Dir, newSize)
    newImg = imresize(img,newSize);
    if exist([Dir filesep 'regionCoords.mat'],'file')
        load([Dir filesep 'regionCoords.mat']);
        curInd = find(strcmp({coord.filename}, fileName)==1);
        if ~isempty(curInd)
            figure(); 
            for index = 1:size(coord(curInd).X,1)
                newY = ([coord(curInd).Y(index,1) coord(curInd).Y(index,2)]/size(img,1))*newSize(1);
                newX = ([coord(curInd).X(index,1) coord(curInd).X(index,2)]/size(img,2))*newSize(2);
                curRegions{index} = newImg(newY(1):newY(2),newX(1):newX(2),:);
                subplot(6,2,2*index-1); imshow(img(coord(curInd).Y(index,1):coord(curInd).Y(index,2),coord(curInd).X(index,1):coord(curInd).X(index,2),:));
                subplot(6,2,2*index); imshow(curRegions{index});
            end
            return;
        else
            curInd = length(coord)+1;
        end
    else
        curInd = 1;
    end
    fh = figure(); imshow(img); hold on
    colorSet = {'r' 'y' 'g' 'b' 'k' 'c'};
    for pointInd = 1:6
        for index = 1:2
            [X(pointInd,index), Y(pointInd,index)]= ginput(1);
            plot(X(pointInd,index),Y(pointInd,index),['.' colorSet{pointInd}]);
        end
        winW = max(X(pointInd,:))-min(X(pointInd,:)); winH = max(Y(pointInd,:))-min(Y(pointInd,:));
        refWin = [min(X(pointInd,:)) min(Y(pointInd,:)) winW winH];
        rectangle('position',refWin, 'edgecolor',colorSet{pointInd}); 
    end
    coord(curInd).filename = fileName;
    coord(curInd).X = X;
    coord(curInd).Y = Y;
%     figure(); 
    for index = 1:6 %size(coord(curInd).X,1)
        newY = ([coord(curInd).Y(index,1) coord(curInd).Y(index,2)]/size(img,1))*newSize(1);
        newX = ([coord(curInd).X(index,1) coord(curInd).X(index,2)]/size(img,2))*newSize(2);
        curRegions{index} = newImg(newY(1):newY(2),newX(1):newX(2),:);
%         subplot(6,2,2*index-1); imshow(img(coord(curInd).Y(index,1):coord(curInd).Y(index,2),coord(curInd).X(index,1):coord(curInd).X(index,2),:));
%         subplot(6,2,2*index); imshow(curRegions{index});
    end
    save([Dir filesep 'regionCoords.mat'],'coord');
    close(fh);
end

function displayRegions(coord, curInd, oldSize, newSize)
    colorSet = {'r' 'y' 'g' 'b' 'k' 'c'};
    for k =1:size(coord(curInd).X,1)
        newY = (coord(curInd).Y(k,:)/oldSize(1))*newSize(1);
        newX = (coord(curInd).X(k,:)/oldSize(2))*newSize(2);
        
        winW = max(newX)-min(newX); winH = max(newY)-min(newY);
        refWin = [min(newX) min(newY) winW winH];
        rectangle('position',refWin, 'edgecolor',colorSet{k},'linewidth',1.5); 
    end
end

%% Divide image to certain number of blocks
function [curRegions] = imageGrid(img, newSize, regionNum)
    % Each matrix defines the number of columns in each row of grid (sum would define the total number of regions)
    blockStruct = {[1 1 1] [2 2] [1 3 1] [1 4 1]}; 
    blockInd = find(cellfun(@(x) sum(x(:)),blockStruct)==regionNum);
    
    newImg = imresize(img,newSize);
    rInd = 1;
    for ii = 1:length(blockStruct{blockInd})
        rowSize = floor(size(newImg,1)/length(blockStruct{blockInd}));
        for jj = 1:blockStruct{blockInd}(ii)
            colSize = floor(size(newImg,2)/blockStruct{blockInd}(ii));
            curRegions{rInd} = newImg((ii-1)*rowSize+1:ii*rowSize, (jj-1)*colSize+1:jj*colSize,:);
            rInd = rInd+1;
        end
    end
end
