% Merge data chuncks from generateData
clear all; clc;
regionNum = 6; 
regionType = 'saliency';
traintestData = [];
minIndStart = 2001; maxIndStart = 380;
% load(['data-' num2str(regionNum) filesep 'fullData-full-' regionType '-' num2str(minIndStart) '00-' num2str(maxIndStart) '00' '.mat']);
minInd = 2001; maxInd = 2461;
for index =minInd:maxInd
    tempdata = load(['data-' num2str(regionNum) filesep 'fullData-full-saliency-' num2str(index) '00.mat']);
    traintestData = vertcat(traintestData,tempdata.traintestData);
    clear tempdata
end
save(['data-' num2str(regionNum) filesep 'fullData-full-' regionType '-' num2str(minIndStart) '00-' num2str(maxInd) '00' '.mat'],'traintestData','-v7.3');