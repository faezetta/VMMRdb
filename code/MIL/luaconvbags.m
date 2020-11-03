%% Convert MIL data to Lua format
function experimentsMIL()
    clear all; clc;
    expDir = '../../vlfeat-0.9.18/apps/recognition';
    expr = 'ex-dbLite16_100-fv-aug';
    type = 'saliency'; %'saliency'; %'manual'; 'grid'
    load(fullfile(expDir,'data',expr,'imdb.mat'));

    regionNum = 6;
    if strcmp(type,'saliency') || strcmp(type,'grid') 
                                  dataDir = ['data-' num2str(regionNum)];
    else                          dataDir = ['data-' type];
    end
    load(fullfile(dataDir,['fullData-full-' type '.mat']));
    trainInd = find(images.set==1);   testInd = find(images.set==3);

    % Train data
    saveData(traintestData, trainInd, 'train');
    saveData(traintestData, testInd, 'test');
    saveData(traintestData, [1:length(images.set)], 'full');
end

% Make sure to save bags and index into a mat file to call from torch
function saveData(data, selectedInd, type)
    makeList = unique({data{:,3}});
    index =[];
    bags = [];
    offset = 1;
    bagList = unique({data{:,2}});
    for ii = 1:length(selectedInd)
        curbagInd = find(strcmp({data{:,2}}, bagList{selectedInd(ii)}));
        for jj=1:length(curbagInd)
            bags = [bags;data{curbagInd(jj),4}];
        end
        curMake = find(strcmp(makeList, data{curbagInd(1),3}));
        index=[index;[curMake offset offset+length(curbagInd)-1]];
        offset = offset+length(curbagInd);
    end
    if ~exist('lua','dir') mkdir('lua');  end
    save(['lua' filesep type '_miltolua.mat'],'bags','index');
end