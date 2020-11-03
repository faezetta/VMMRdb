function compareMisClassifiedBags()
    close all; clear all; clc;
    %% Input data
    addpath(genpath('..\..\vlfeat-0.9.18\toolbox'));   
    expDir = '..\..\vlfeat-0.9.18\apps\recognition';
    expr = 'ex-dbLite16_100-fv-aug';
    type = 'saliency'; 
    
    load(fullfile('data-all',['fullData-full-' type '.mat']));
    load(fullfile(expDir,'data',expr,'imdb.mat'));
    load(fullfile(expDir,'data',expr,'result.mat'));
    curRun = load(['Results' filesep 'Run-saliency-all_inst_MI_SVM -Kernel 0.mat']);
    load([expDir filesep imageDir filesep 'regionCoordsSaliency-org.mat']);
    
    makeList = unique({traintestData{:,3}});
    testInd = find(images.set==3);
    imgDir = [expDir filesep imageDir];
    instProbThr = 0.4;
    
    destDir = 'MIL_misclassified';
%     if exist(destDir, 'dir')
%         rmSubDir(destDir);
%     end
%     mkdir(destDir);
%     
    % Define testset instances
    startPos = find(strcmp('inst-1', {traintestData{:,1}}));
    testInst = [];
    for index=1:length(testInd)
        testInst = [testInst; ones(length(startPos(testInd(index)):startPos(testInd(index)+1)-1),1).*index];
    end
    
    misType = 'MIL'; %'SIL'
    kk = 0;
    
    [~,SILpreds] = max(scores, [], 1);
    for index = 1:length(makeList)
        sel = find(images.class == index & images.set == 3);
        if strcmp(misType,'SIL')
            % SIL misclassified images
            misInd{index} = find(SILpreds(sel)~=index);
            curMisInd = sel(misInd{index});

            % MIL label
            [~,testmisInd]=ismember(curMisInd,testInd);
            curBagLbl = curRun.run{index}.bag_label(testmisInd);
        else
            instProbThr = 0.0;
            [~,curtestInd]=intersect(testInd,sel);
            curBagLbl = curRun.run{index}.bag_label(curtestInd);
            misIndNum{index} = find(curBagLbl==0);
            misInd = find(curBagLbl==0);
            curMisInd = testInd(curtestInd(misInd));
            for ii=1:length(misInd)
                if ~exist([destDir filesep makeList{index}], 'dir')
                    mkdir([destDir filesep makeList{index}]);
                end
                copyfile([expDir filesep imageDir filesep images.name{curMisInd(ii)}],...
                         [destDir filesep images.name{curMisInd(ii)}]);
                kk = kk+1;
                milMis(kk).filename = images.name{curMisInd(ii)};
                milMis(kk).curProb = curRun.run{index}.bag_prob(curtestInd(misInd(ii)));
                milMis(kk).allProb = cellfun(@(x)(x.bag_prob(curtestInd(misInd(ii)))), curRun.run);
            end
        end
        % instances
        load(['data-all' filesep 'data_' makeList{index} '_test_0_nans.mat']);
        deletedInds  = find(nanInd~=0);
        newInstLb = curRun.run{index}.inst_label;
        confInst = curRun.run{index}.inst_prob;
        for ii=1:length(deletedInds)
            newInstLb = [newInstLb(1:deletedInds(ii)-1);0;newInstLb(deletedInds(ii):end)];
            confInst = [confInst(1:deletedInds(ii)-1);0;confInst(deletedInds(ii):end)];
        end

%         figure('Name', makeList{index}, 'units','normalized','outerposition',[0 0 1 1],'color','w'); clf();
%         ha = tight_subplot(3,5,[.04 .04],[.04 .04],[.01 .01]);
%         for ii=1:length(curMisInd)
%             axes(ha(ii)); cla();
%             curImg = imread([imgDir filesep images.name{curMisInd(ii)}]);
%             imshow(curImg);
%             instInd = find(testInst==find(testInd==curMisInd(ii)));
%             hold on
%             for jj=1:length(instInd)
%                 if newInstLb(instInd(jj))==1 bbCol = 'r';
%                 else                         bbCol = [0.95 0.95 0.95];
%                 end
%                 if confInst(instInd(jj))>instProbThr
%                     rectangle('Position',coord(curMisInd(ii)).bb(jj,:),'EdgeColor',bbCol,'lineWidth',1.5);
%                     text(coord(curMisInd(ii)).bb(jj,1),coord(curMisInd(ii)).bb(jj,2),num2str(confInst(instInd(jj)),'%.2f'),...
%                              'BackGroundColor','w','FontWeight','b','fontSize',9.0);
%                 end
%             end
%             hold off
%             if strcmp(misType,'SIL')
%                 misMake = strrep(strtok(makeList{SILpreds(curMisInd(ii))},'\'),'_',' ');
%                 title({['SIL: ' misMake],['MIL: Bag label ' num2str(curBagLbl(ii))]},'fontweight','b','fontSize',9.5);
%             end
%         end
        clear sel testmisInd curBagLbl
    end
    save([destDir,filesep,'milMisProb.mat'],'milMis');
end

%
function rmSubDir(pathDir)
    d = dir(pathDir);
    isub = [d(:).isdir];
    nameFolds = {d(isub).name}';
    nameFolds(ismember(nameFolds,{'.','..'})) = [];

     for i=1:size(nameFolds,1)
        dir2rm = fullfile(pathDir,nameFolds{i});
        rmdir(dir2rm, 's');
     end
end
