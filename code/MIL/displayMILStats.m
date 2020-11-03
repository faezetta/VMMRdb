function displayMILStats()
    clear all; clc;
    workspace;
    addpath(genpath('..\..\vlfeat-0.9.18\toolbox'));   
    expType = 'SILvs.MIL'; % {'MIL-manual-multi','MIL-topinstances','SILvs.MIL','MIL-multi'};
  
    %% Input data
    type = 'saliency'; %'manual';
    load(fullfile('data-all',['fullData-full-' type '.mat']));
    makeList = unique({traintestData{:,3}});
    expDir = '..\..\vlfeat-0.9.18\apps\recognition';
    expr = 'ex-dbLite16_100-fv-aug';
    load(fullfile(expDir,'data',expr,'imdb.mat'));
    testInd = find(images.set==3);
    
    %% SIL 
    confusion = [];
    silDir = '..\..\vlfeat-0.9.18\apps\recognition\data\ex-dbLite16_100-fv-aug';
    load([silDir filesep 'result.mat']); %'result-testROI.mat'   'result-full.mat'
    SILAcc = 100.*diag(confusion);

    if strcmp(expType,'MIL-manual-multi')
        %% Manual MIL - Instance-level results
        load(fullfile('data-manual',['fullData-full-manual.mat']));
        makeList = unique({traintestData{:,3}});
        %  Display the frequency of accurate labels per class
        instNum = [2 3 5 6];
        testImgNum = 10;
        instName = {'LTLight', 'RTLight', 'Bumper', 'LBB', 'RBB', 'LPArea'};
        manualBagAcc = zeros(length(makeList),length(instNum));
        for instInd = 1:length(instNum)
            load(['Results' filesep 'Run-' num2str(instNum(instInd)) '_inst_MI_SVM -Kernel 0.mat']);
            milLbl{instInd} = ['MIL-' num2str(instNum(instInd))];
            figure(1); set(1,'units','normalized','outerposition',[0 0 1 1],'color','w');   
            ha = tight_subplot(length(makeList)/2,2,[.06 .05],[.05 .03],[.04 .04]);
            for index = 1:length(makeList)
                manualBagAcc(index,instInd) = run{index}.BagAccu*100;
                load(['data-manual' filesep 'data_' makeList{index} '_test_' num2str(instNum(instInd)) '_nans.mat']);
                deletedInds  = find(nanInd~=0);
                newInstLb = run{index}.inst_label;
                for ii=1:length(deletedInds)
                    newInstLb = [newInstLb(1:deletedInds(ii)-1);0;newInstLb(deletedInds(ii):end)];
                end
                run{index}.inst_label = newInstLb;
                curmakelabels = reshape(run{index}.inst_label,instNum(instInd),length(testInd))';
                for ii=1:length(makeList)
                    templabels = curmakelabels((ii-1)*testImgNum+1:ii*testImgNum,:);
                    labelfreq(ii,:) = arrayfun(@(j) histc(templabels(:,j), 1), 1:size(templabels,2) );
                end
                axes(ha(index));
                bh = bar(labelfreq,'EdgeColor',[0.4 0.4 0.4],'LineWidth',0.5); ylabel('Freq. pos labels');
                set(bh(1),'FaceColor',[0.8 0 0]);
                set(bh(2),'FaceColor',[1.0 0.4 0.4]);
                if instNum(instInd)>2 set(bh(3),'FaceColor',[1.0 1.0 0.0]);end
                if index==2 legend(bh,{instName{1:instNum(instInd)}}, 'Location','NorthEast','FontSize',7); end
                curMake = strrep(makeList{index},'_',' '); ht = title(curMake,'FontWeight', 'b');
                set(gca,'ylim',[0 testImgNum],'xtick',[1:length(makeList)],'XGrid','off','YGrid','on','fontSize',7.5,'box','on'); 

                clear newInstLb curmakelabels templabels labelfreq
            end
            vl_printsize(1);
            print('-dmeta', fullfile(pwd, 'data-manual', ['instanceLblDist-' num2str(instNum(instInd)) '.emf'])) ;
        end
    elseif strcmp(expType,'MIL-topinstances')
        %% Saliency MIL - Instance level
        %  Visualization of instance labels on top bags per class
        instNum = 0;
        topNum = 15;
        instprobThr = 0.0;
        curRun = load(['Results' filesep 'Run-saliency-all_inst_MI_SVM -Kernel 0.mat']);
        imageDir = '..\..\vlfeat-0.9.18\apps\recognition\data\dbLite16_100';

        curRunCKNN = load(['Results' filesep 'Run-saliency-all_kNN –RefNum 5 –CiterRank 5.mat']);
        bagAccCKNN = [cellfun( @(x) x.BagAccu, curRunCKNN.run )]'.*100;

        load([imageDir filesep 'regionCoordsSaliency-org.mat']);
        avgInstNum = mean(arrayfun(@(s) size(s.bb,1), coord))
        avgInstNumTest = mean(arrayfun(@(s) size(s.bb,1), coord(testInd)))

        bagAcc = [cellfun( @(x) x.BagAccu, curRun.run )]'.*100;
        % Defeine testset instances
        startPos = find(strcmp('inst-1', {traintestData{:,1}}));
        testInst = [];
        for index=1:length(testInd)
            testInst = [testInst; ones(length(startPos(testInd(index)):startPos(testInd(index)+1)-1),1).*index];
        end

        for index = 1:length(makeList)
            load(['data-all' filesep 'data_' makeList{index} '_test_' num2str(instNum) '_nans.mat']);

            deletedInds  = find(nanInd~=0);
            newInstLb = curRun.run{index}.inst_label;
            confInst = curRun.run{index}.inst_prob;
            for ii=1:length(deletedInds)
                newInstLb = [newInstLb(1:deletedInds(ii)-1);0;newInstLb(deletedInds(ii):end)];
                confInst = [confInst(1:deletedInds(ii)-1);0;confInst(deletedInds(ii):end)];
            end
            curRun.run{index}.inst_label = newInstLb;
            curRun.run{index}.inst_prob = confInst;

            figure(1); set(1,'Name',makeList{index},'units','normalized','outerposition',[0 0 1 1],'color','w'); clf();
            ha = tight_subplot(3,topNum/3,[.02 .02],[.02 .02],[.02 .02]);
            [bpVal, bpInd] = sort(curRun.run{index}.bag_prob,'descend');
            for ii=1:topNum
                axes(ha(ii)); cla();
                curImg = imread([imageDir filesep coord(testInd(bpInd(ii))).filename]); imshow(curImg);
                curMake = strrep(strtok(coord(testInd(bpInd(ii))).filename,'\'),'_','');
                title(['Bag prob.: ' num2str(bpVal(ii)) ' - Bag label: ' num2str(curRun.run{index}.bag_label(bpInd(ii))) ' - ' curMake],...
                      'fontweight','b','fontSize',9);
                instInd = find(testInst==bpInd(ii));
                for jj=1:length(instInd)
                    curInstProb = curRun.run{index}.inst_prob(instInd(jj));
                    if curInstProb>instprobThr
                        if curRun.run{index}.inst_label(instInd(jj))==1 bbCol = 'g';
                        else                                     bbCol = 'r';
                        end
                        rectangle('Position',coord(testInd(bpInd(ii))).bb(jj,:),'EdgeColor',bbCol,'lineWidth',1.5);
                        text(coord(testInd(bpInd(ii))).bb(jj,1),coord(testInd(bpInd(ii))).bb(jj,2),num2str(curInstProb),...
                             'BackGroundColor','w','FontWeight','b','fontSize',7.5);
                    end
                end
            end
            vl_printsize(1);
            print('-dmeta', fullfile(pwd, 'data', ['regionResults-' makeList{index} '.emf'])) ;
        end
    elseif strcmp(expType,'SILvs.MIL')
        %% MIL
        curRun = load(['Results' filesep 'Run-saliency-all_inst_MI_SVM -Kernel 0.mat']);
        bagAcc = [cellfun( @(x) x.BagAccu, curRun.run )]'.*100;
        
        curRunCKNN = load(['Results' filesep 'Run-saliency-all_kNN –RefNum 5 –CiterRank 5.mat']);
        bagAccCKNN = [cellfun( @(x) x.BagAccu, curRunCKNN.run )]'.*100;
        
        makeList = cellfun(@(x) strrep(x,'_',''),makeList,'Un',0);

        figure(2); set(2,'color','w','units','normalized','outerposition',[0 0 1 1]); clf ;   set(gcf, 'color', 'none');
        bh = bar([SILAcc bagAcc bagAccCKNN],'LineWidth',0.5,'EdgeColor',[0.3,0.3,0.3]); colormap gray
        set(bh(1),'FaceColor',[0.47 0.2 0.42]);
        set(bh(2),'FaceColor',[0.65 0.65 0.65], 'EdgeColor', [.7 .7 .7]);
        set(bh(3),'FaceColor',[0.25 0.25 0.25], 'EdgeColor', [.7 .7 .7]);
        meanAccuracy = sprintf('SIL mean accuracy: %.2f\n Saliency MIL mean accuracy (MI-SVM): %.2f - Saliency MIL mean accuracy (CKNN): %.2f\n', ...
                                mean(SILAcc), mean(bagAcc), mean(bagAccCKNN));
        title(meanAccuracy);  
        set(gca,'Fontname','Times','fontSize',16);
        ylabel('Accuracy %');  set(gca,'fontweight','b', 'xlim', [0 length(makeList)+1], 'xtick', [1:length(makeList)]);
        set(gca,'XTick',1:length(makeList),'XTickLabel',makeList,'TickLength',[0 0],'fontSize',16,'Fontname','Times','YColor','k','XColor','k');
        rotateTickLabel(gca,17,1); 
        ylim([0 100]);     %grid on;
        bhl = legend(bh, {'SI-SVM','MI-SVM^{s}','CkNN^{s}'}, 'Location','SouthEastOutside','FontSize',15,'color','k');
        set(gca, 'color', 'none'); set(bhl,'color','none','EdgeColor',[.7 .7 .7]);
        hText = findobj(bhl, 'type', 'text');
        set(hText,'color', 'k');
        
%         export_fig perfect_solution.png

        vl_printsize(2) ;
        print('-dmeta', fullfile(pwd, 'data', ['result-acc.emf'])) ;
    elseif strcmp(expType,'MIL-multi')
        makeList = cellfun(@(x) strrep(x,'_',''),makeList,'Un',0);
        regionNum = [3 4 5 6];
        meanAccuracy = {'' ''}; legStr={};
        bagAcc = zeros(length(makeList),2*length(regionNum));
        % Grid
        for index = 1:length(regionNum)
            if exist(['Results' filesep 'Run-grid-' num2str(regionNum(index)) '_inst_MI_SVM -Kernel 0.mat'],'file')==2
                curRun = load(['Results' filesep 'Run-grid-' num2str(regionNum(index)) '_inst_MI_SVM -Kernel 0.mat']);
                bagAcc(:,index) = [cellfun( @(x) x.BagAccu, curRun.run )]'.*100;
            end
            meanAcc(index) = mean(bagAcc(:,index));
            meanAccuracy{1} = [meanAccuracy{1}  '   ' num2str(regionNum(index)) '-Grid Region mean acc.: ', num2str(meanAcc(index),4)];
            legStr{index} = ['MI-SVM-Grid-' num2str(regionNum(index))]; 
        end
        % Saliency
        for index = 1:length(regionNum)
            if exist(['Results' filesep 'Run-saliency-' num2str(regionNum(index)) '_inst_MI_SVM -Kernel 0.mat'],'file')==2
                curRun = load(['Results' filesep 'Run-saliency-' num2str(regionNum(index)) '_inst_MI_SVM -Kernel 0.mat']);
                bagAcc(:,index+length(regionNum)) = [cellfun( @(x) x.BagAccu, curRun.run )]'.*100;
            end
            meanAcc(index) = mean(bagAcc(:,index+length(regionNum)));
            meanAccuracy{2} = [meanAccuracy{2} '   ' num2str(regionNum(index)) '-Salient Region mean acc.: ', num2str(meanAcc(index),4)];
            legStr{index+length(regionNum)} = ['MI-SVM-Salient-' num2str(regionNum(index))]; 
        end
        figure(3); set(3,'color','w','units','normalized','outerposition',[0 0 1 1]); clf ; 
        bh = bar(bagAcc,'LineWidth',0.5,'EdgeColor',[0.3,0.3,0.3]); colormap gray
        for index=1:length(regionNum)
            set(bh(index),'FaceColor',[0.75/index 0 0]);
        end
        title(meanAccuracy);  
        ylabel('Accuracy %');  set(gca,'fontweight','b', 'xlim', [0 length(makeList)+1], 'xtick', [1:length(makeList)]);
        set(gca,'XTick',1:length(makeList),'XTickLabel',makeList,'TickLength',[0 0],'fontSize',9.5,'Fontname','Timesnewroman');
        rotateTickLabel(gca,20,1);
        ylim([0 100]);     grid on;
        legend(bh, legStr, 'Location','SouthEast','FontSize',9.5);
        
        meanAcc = reshape(mean(bagAcc),[],2)';
        figure(4); set(4,'color','w'); clf ; 
        lineCol = ['r' 'k'];
        hold on
        for index = 1:2
            plot([1:length(regionNum)],meanAcc(index,:),['-.' lineCol(index) 's'],...
                 'LineWidth',2,'MarkerSize',10,'MarkerEdgeColor',[0.5,0.5,0.5],'MarkerFaceColor',lineCol(index));
        end
        hold off
        set(gca,'XTick',1:length(regionNum),'XTickLabel',regionNum,'TickLength',[0 0],'fontSize',9.5,'Fontname','Timesnewroman','fontweight','b');
        ylabel('Average accuracy %');  xlabel('Number of regions');  grid on;
        legend({'Fixed grid regions','Salient regions'}, 'Location','SouthEast','FontSize',9.5);
    end
end

%% Fixing data 
%     ind=find(ismember(images.name,'Toyota_Corolla_2006-2010\Toyota_Corolla_2010_3 - Copy.jpg'));
%     for index = 1:length(ind)
%         images.name{ind(index)} = 'Toyota_Corolla_2006-2010\Toyota_Corolla_2010_33.jpg';
%     end 

%     ind=find(ismember({traintestDataTmp{:,2}},'Toyota_Corolla_2010_33 (2).jpg'));
%     [traintestDataTmp{ind,2}] = deal('Toyota_Corolla_2010_34.jpg');
%     save([dataDir filesep 'fullData.mat'],'traintestDataTmp','-v7.3');

%     index = find(strcmp({coord.filename}, 'Toyota_Corolla_2006-2010\Toyota_Corolla_2010_33 (2).jpg')==1);
%     coord(index).filename = 'Toyota_Corolla_2006-2010\Toyota_Corolla_2010_34.jpg';
%     save([imageDir filesep 'regionCoordsSaliency.mat'],'coord');
