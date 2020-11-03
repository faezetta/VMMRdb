function retrieveTopInstances()
    close all; clear all; clc;
    %% Input data
    addpath(genpath('..\..\vlfeat-0.9.18\toolbox'));   
    expDir = '..\..\vlfeat-0.9.18\apps\recognition';
    expr = 'ex-dbLite16_100-fv-aug';
    type = 'saliency'; 
    
    load(fullfile('data-all',['fullData-full-' type '.mat']));
    load(fullfile(expDir,'data',expr,'imdb.mat'));
    curRun = load(['Results' filesep 'Run-saliency-all_inst_MI_SVM -Kernel 0.mat']);
    load([expDir filesep imageDir filesep 'regionCoordsSaliency-org.mat']);
    
    makeList = unique({traintestData{:,3}});
    testInd = find(images.set==3);
    trainInd = find(images.set==1);
    imgDir = [expDir filesep imageDir];
    
    topNum = 42;
    instprobThr = 0.0;
    
    % Define testset instances
    startPos = find(strcmp('inst-1', {traintestData{:,1}}));
    testInst = [];
    for index=1:length(testInd)
        testInst = [testInst; ones(length(startPos(testInd(index)):startPos(testInd(index)+1)-1),1).*index];
    end
    
    for index = 1:length(makeList)
        load(['data-all' filesep 'data_' makeList{index} '_test_0_nans.mat']);
        
        deletedInds  = find(nanInd~=0);
        newInstLb = curRun.run{index}.inst_label;
        confInst = curRun.run{index}.inst_prob;
        for ii=1:length(deletedInds)
            newInstLb = [newInstLb(1:deletedInds(ii)-1);0;newInstLb(deletedInds(ii):end)];
            confInst = [confInst(1:deletedInds(ii)-1);0;confInst(deletedInds(ii):end)];
        end
        curRun.run{index}.inst_label = newInstLb;
        curRun.run{index}.inst_prob = confInst;
        
        figure('Name', makeList{index}, 'units','normalized','outerposition',[0 0 1 1],'color','w'); clf();
        ha = tight_subplot(7,6,[.03 .03],[.02 .02],[.02 .02]);
        [bpVal, bpInd] = sort(curRun.run{index}.bag_prob,'descend');
        [instpVal, instpInd] = sort(curRun.run{index}.inst_prob,'descend');
        ii = 1; kk=1;
        while kk<=topNum
            axes(ha(kk)); cla();
            curImg = imread([imgDir filesep coord(testInd(testInst(instpInd(ii)))).filename]); imshow(curImg);
            curMake = strrep(strtok(coord(testInd(testInst(instpInd(ii)))).filename,'\'),'_','');
            title(['Bag prob.: ' num2str(curRun.run{index}.bag_prob(testInst(instpInd(ii))),'%.2f') ...
                   ' - ' curMake],... %' - Bag lbl: ' num2str(curRun.run{index}.bag_label(testInst(instpInd(ii))),'%.4f') 
                   'fontweight','b','fontSize',9);
            bestInstProb = instpVal(ii); %curRun.run{index}.inst_prob(instInd(jj));
            
            instInd = find(testInst==testInst(instpInd(ii)));
            for jj=1:length(instInd)
                curInstProb = curRun.run{index}.inst_prob(instInd(jj));
                if instpInd(ii)==instInd(jj) && prod(coord(testInd(testInst(instpInd(ii)))).bb(jj,3:4))<(0.5)*prod([size(curImg,1) size(curImg,2)])
                    bbCol = 'r';
                    rectangle('Position',coord(testInd(testInst(instpInd(ii)))).bb(jj,:),'EdgeColor',bbCol,'lineWidth',1.5);
%                     text(coord(testInd(testInst(instpInd(ii)))).bb(jj,1),coord(testInd(testInst(instpInd(ii)))).bb(jj,2),num2str(curInstProb),...
%                          'BackGroundColor','w','FontWeight','b','fontSize',7.5);
                    kk = kk+1;
                end
            end
            ii = ii+1;
        end
    %         vl_printsize(1);
%         print('-dmeta', fullfile(pwd, 'data', ['regionResults-' makeList{index} '.emf'])) ;
    end
end