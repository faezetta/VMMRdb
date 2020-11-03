%% "Saliency Detection: A Boolean Map Approach", Jianming Zhang, Stan Sclaroff, ICCV, 2013  
%  sod: boolean value for using the salient object detection mode
function [saliencyRegions] = saliencyBMS_MILCNN(curImg, cur_filename, Dir, regionNum)
    addpath(genpath('Saliency\BMS-mex-v2'));
%     addpath(genpath('Saliency\BMS-mex'));
    imSize = [224 224];
    input_dir           =   'tempin/';
    output_dir          =   'tempout/';
    if ~exist(input_dir,'dir') mkdir(input_dir);  end
    imwrite(curImg,[input_dir filesep 'curImg.jpg']);
    if ~exist(output_dir,'dir') mkdir(output_dir);  end
    
    sample_step_size    =   8;   % \delta
    max_dim             =   450; % maximum dimension of the image
    % do not change the following
    dilation_width_1    =   max(round(7*max_dim/400),1); % \omega
    dilation_width_2    =   max(round(9*max_dim/400),1); % \kappa
    blur_std            =   round(9*max_dim/400); % \sigma
    color_space         =   2;   % RGB: 1; Lab: 2; LUV: 4
    whitening           =   1;   % do color whitening
    mexBMS(input_dir,output_dir,sample_step_size,dilation_width_1,dilation_width_2,blur_std,color_space,whitening,max_dim);
    movefile([output_dir filesep 'curImg.png'],[output_dir filesep 'curImg-eye.png']);
    I_eye               =   imread([output_dir filesep 'curImg-eye.png']);

    % for salient object detection
%     sample_step_size    =   8;   % \delta
%     opening_width       =   13;  % \omega_o
%     dilation_width_1    =   1;   % \omega_{d1} (turn off dilation)
%     dilation_width_2    =   1;   % \omega_{d2} (turn off dilation)
%     blur_std            =   0;   % \sigma (turn off bluring)
%     use_normalization   =   0;   % L2-normalization
%     handle_border       =   0;
%     mexBMSSalient(input_dir,output_dir,sample_step_size,opening_width,dilation_width_1,dilation_width_2,blur_std,use_normalization,handle_border);
    % post-processing for salient object detection tasks
    radius              =   15;
    I                   =   I_eye; %imread([output_dir 'curImg.png']);
    I                   =   postProcSOD(I,radius);
    imwrite(I,fullfile(output_dir,['curImg.png']));
    
    saliencyRegions = extractRegions(curImg, cur_filename, I_eye, I, Dir, regionNum, imSize);
       
    delete([input_dir filesep 'curImg.jpg']);
    delete([output_dir filesep 'curImg.png']);
end

%% Post-processing function for Salient Object Detection
function sMap           =   postProcSOD(mAttMap,radius)
    % opening by reconstruction followed by closing by reconstruction (watershed/ipexwatershed)
    img_size            =   size(mAttMap);
    I                   =   imresize(mAttMap,[NaN 450]); %  width = 400
    se                  =   strel('disk',radius);
    Ie                  =   imerode(I, se);
    Iobr                =   imreconstruct(Ie, I);
    Iobrd               =   imdilate(Iobr, se);
    Iobrcbr             =   imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
    Iobrcbr             =   imcomplement(Iobrcbr);
    sMap                =   imresize(mat2gray(Iobrcbr),img_size(1:2));
end

%% Extract regions based on the areas selected by eye fixation and saliency detection
function [curRegions] = extractRegions(curImg, cur_filename, I_eye, I, Dir, regionNum, imSize)
    curInd = 1;  
    regions = [];

    eyeThr              =   [100 95  90   85   80  75  70   65   60  55  50   45   40   35   30   25   20   15   10   5];  
    saliencyThr         =   [0.4 0.4 0.35 0.35 0.3 0.3 0.25 0.25 0.2 0.2 0.15 0.15 0.10 0.10 0.05 0.05 0.05 0.02 0.02 0.01]; 
    bbRatio             =   1/3;
    bbRatio2            =   1/2;
    sizeThr             =   50;
    [eyeInd, salInd]    =   deal(1);
    
    while (size(regions,1)~=regionNum)
        I1 = zeros(size(I_eye)); I1(find(I_eye>eyeThr(eyeInd)))=1; 
        [regions1, I1] = refineRegions(I1);

        I2 = zeros(size(I));     I2(find(I>=saliencyThr(salInd)))=1;
        I3 = imcomplement(I1).*I2;
        [regions2,~] = refineRegions(I3);
        regions = [regions1;regions2];
        
        % Vary thresholds to increase/decrease the number of regions
        if size(regions,1)>regionNum
            [curEye, curSaliency] = deal([]);
            for index=1:size(regions1,1)
                curEye(index) = mean2(imcrop(I1,regions1(index,:)));
            end
            for index=1:size(regions2,1)
                curSaliency(index) = mean2(imcrop(I2,regions2(index,:)));
            end
            curVal = [curEye curSaliency];
            [~, regionInd] = sort(curVal, 'descend' );
            regions = regions(regionInd(1:regionNum),:);
        else
            if (eyeInd==salInd)  eyeInd = eyeInd+1;
            else                 salInd = salInd+1;
            end
            % In case the number of regions is not achieved, use duplicate regions
            if eyeInd>length(eyeThr) || salInd>length(saliencyThr)
                curRegionNum = regionNum-1;
                while curRegionNum~=0
                    [eyeInd, salInd] = deal(1);
                    for index = 1:length(eyeThr)
                        I1 = zeros(size(I_eye)); I1(find(I_eye>eyeThr(eyeInd)))=1; 
                        [regions1, I1] = refineRegions(I1);
                        I2 = zeros(size(I));     I2(find(I>=saliencyThr(salInd)))=1;
                        I3 = imcomplement(I1).*I2;
                        [regions2,~] = refineRegions(I3);
                        if size(regions1,1)==0 && size(regions2,1)==0
                            regions = [regions;regions1;regions2];
                        else
                            regions = [regions1;regions2];
                        end
                        if size(regions,1)==curRegionNum
%                             for ii=1:(regionNum-curRegionNum)
%                                 curRow = randi([1 curRegionNum],1,1);
%                                 regions = [regions;regions(curRow,:)];
%                             end
                            regionNum = size(regions,1);
                            break;
                        end
                    end
                    if size(regions,1)==regionNum
                        break;
                    else
                        curRegionNum = curRegionNum-1;
                    end
                end
            end
        end
    end
       
%     figure('units','normalized','outerposition',[0 0 1 1],'color','w'); 
%     subplot(1,3,1); imshow(I_eye);     subplot(1,3,2); imshow(I);     
%     subplot(1,3,3); imshow(curImg); hold on; 
    for index = 1:size(regions,1)
        curBB = regions(index,:);
        curBBExpanded(index,:) = [curBB(1)-curBB(3)*bbRatio/2 curBB(2)-curBB(4)*bbRatio/2 (1+bbRatio)*curBB(3) (1+bbRatio)*curBB(4)];
        if curBBExpanded(index,3)<sizeThr || curBBExpanded(index,4)<sizeThr
            curBBExpanded(index,:) = [curBB(1)-curBB(3)*bbRatio2/2 curBB(2)-curBB(4)*bbRatio2/2 (1+bbRatio2)*curBB(3) (1+bbRatio2)*curBB(4)];
        end
        % make it square
        maxDim = max(curBBExpanded(index,3),curBBExpanded(index,4));
        curBBExpanded(index,:) = [curBB(1)-(maxDim-curBBExpanded(index,3))/2 curBB(2)-(maxDim-curBBExpanded(index,4))/2 maxDim maxDim];
        
        curRegions{index} = imcrop(curImg,curBBExpanded(index,:));
        imwrite(imresize(curRegions{index},imSize),[Dir filesep cur_filename(1:end-4) '_' num2str(index) '.jpg']);
    end
    % adding the overall boundingbox
    mainbb = [min(curBBExpanded(:,1)), min(curBBExpanded(:,2)),...
              max(curBBExpanded(:,1)+curBBExpanded(:,3))-min(curBBExpanded(:,1)),...
              max(curBBExpanded(:,2)+curBBExpanded(:,4))-min(curBBExpanded(:,2))];
    curRegions{index+1} = imcrop(curImg,mainbb);
    imwrite(imresize(curRegions{index+1},imSize),[Dir filesep cur_filename(1:end-4) '_' num2str(index+1) '.jpg']);
%     hold off
end

function [regions, input_morph] = refineRegions(input)
    regionSizeThr       =   [1*min(size(input))/3 100000];
    radius              =   3;
    
    se                  =   strel('disk',radius);
    input_morph         =   imerode(input, se);
    [input_label,~]     =   bwlabel(input_morph);
    stats               =   regionprops(input_label, {'Area', 'BoundingBox'});
    idx = find((regionSizeThr(1) <= [stats.Area]) & ([stats.Area] <= regionSizeThr(2)));
    regions = cell2mat({stats(idx).BoundingBox}');
end
