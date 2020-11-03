%% Generate data in MILL format for instances of all train and test data 
function generateData(expr, expDir, type, regionNum, minInd, maxInd)   
    imSize = [450 600];
    edgeFilter = 0;
    
    % Get image database
    load(fullfile(expDir,'data',expr,'imdb.mat'));
    encoder = load(fullfile(expDir,'data',expr,'encoder.mat'));
    encoderCnt = encoder.means;
    
    traintestData = []; curInd = [];
    if exist(['data-' num2str(regionNum) filesep 'fullData-full-' type '-' num2str(minInd+500-1) '.mat'],'file')
        load(['data-' num2str(regionNum) filesep 'fullData-full-' type '-' num2str(minInd+500-1) '.mat']);
    end
    for i=minInd:maxInd %1:length(images.name)
        cur_filename = images.name{i};
        fileParts = regexp(cur_filename, filesep, 'split');
        if ~isempty(traintestData)
            curInd = strfind([traintestData{:}], fileParts{2});           
        end
        disp(['Extracting instances for: {image id:' num2str(i) '  ' cur_filename '}']);
        if isempty(curInd)
            curImg = imread([imageDir filesep cur_filename]);
            if strcmp(type,'saliency')
                curRegions = saliencyBMS(curImg, cur_filename, ['data-' num2str(regionNum)], regionNum, maxInd);
            elseif strcmp(type,'grid')
                curRegions = imageGrid(curImg, imSize, regionNum);
            else
                curRegions = extractRegions(curImg, cur_filename, [expDir filesep imageDir], imSize);
            end
            data(i).FV = encodeOne(encoder, curRegions, edgeFilter);
            data(i).make = fileParts{1};
            data(i).fileName = fileParts{2};
            traintestData = fixDataFormat(traintestData, data(i), type);
            if ~exist(['data-' num2str(regionNum)], 'dir')
                mkdir(['data-' num2str(regionNum)]);
            end
            if ((mod(i,100)==0) || (i==maxInd))
                save(['data-' num2str(regionNum) filesep 'fullData-full-' type '-' num2str(i) '.mat'],'traintestData');
                traintestData = [];
            end
%             close all;
        end
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

%% Extract Fishervector for test data
function [fvdesc] = encodeOne(encoder, curRegions, edgeFilter)
    for index=1:length(curRegions)
        im = customresize( curRegions{index}, size(curRegions{index},1), size(curRegions{index},2));
        features = encoder.extractorFn(im) ;
        % Keep features from pixels close to edges
        if edgeFilter
            marginVal = 2;
            features = filterEdgeFeatures(im, features, marginVal);
        end
        imageSize = size(im) ;
        psi = {} ;
        for i = 1:size(encoder.subdivisions,2)
            minx = encoder.subdivisions(1,i) * imageSize(2) ;
            miny = encoder.subdivisions(2,i) * imageSize(1) ;
            maxx = encoder.subdivisions(3,i) * imageSize(2) ;
            maxy = encoder.subdivisions(4,i) * imageSize(1) ;
            ok = ...
            minx <= features.frame(1,:) & features.frame(1,:) < maxx  & ...
            miny <= features.frame(2,:) & features.frame(2,:) < maxy ;
            descrs{i} = encoder.projection * bsxfun(@minus, ...
                                               features.descr(:,ok), ...
                                               encoder.projectionCenter) ;
            if encoder.renormalize
                descrs{i} = bsxfun(@times, descrs{i}, 1./max(1e-12, sqrt(sum(descrs{i}.^2)))) ;
            end
            w = size(im,2) ;
            h = size(im,1) ;
            framesinit{i} = features.frame(1:2,:) ;
            frames{i} = bsxfun(@times, bsxfun(@minus, framesinit{i}, [w;h]/2), 1./[w;h]) ;
            descrs{i} = extendDescriptorsWithGeometry(encoder.geometricExtension, frames, descrs{i}) ;
            switch encoder.type 
                case 'fv'
                    z = vl_fisher(descrs{i}, ...
                                  encoder.means, ...
                                  encoder.covariances, ...
                                  encoder.priors, ...
                                  'Improved');%'Improved') ; 'Fast'
                case 'vlad'
                    [words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
                                                       descrs{i}, ...
                                                       'MaxComparisons', 15) ;
                    assign = zeros(encoder.numWords, numel(words), 'single') ;
                    assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
                    z = vl_vlad(descrs{i}, ...
                                encoder.words, ...
                                assign, ...
                                'SquareRoot', ...
                                'NormalizeComponents') ;
            end
            z = z / max(sqrt(sum(z.^2)), 1e-12) ;
            psi{i} = z(:) ;
        end
        fvdesctmp{index} = cat(1, psi{:}) ;
%         siftdescrs{index} = descrs;
%         siftpos{index} = framesinit;
    end
    fvdesc = cell2mat(fvdesctmp)';
end

%% Convert data to MIL format
function [traintestData] = fixDataFormat(traintestData, data, type)
    instanceLabels = {'LTLight', 'RTLight', 'Bumper', 'LBB', 'RBB', 'LPArea'};
    k = size(traintestData,1);
    for index = 1:size(data.FV,1)
        if strcmp(type,'manual')
            traintestData{k+index,1} = instanceLabels{index};
        else
            traintestData{k+index,1} = ['inst-' num2str(index)];
        end
        traintestData{k+index,2} = data.fileName;
        traintestData{k+index,3} = data.make;
        traintestData{k+index,4} = data.FV(index,:);
    end
end