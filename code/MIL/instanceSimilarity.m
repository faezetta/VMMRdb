function varargout = instanceSimilarity(varargin)
    % INSTANCESIMILARITY MATLAB code for instanceSimilarity.fig
    % Edit the above text to modify the response to help instanceSimilarity

    % Last Modified by GUIDE v2.5 10-Mar-2016 16:13:28

    % Begin initialization code - DO NOT EDIT
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
                       'gui_Singleton',  gui_Singleton, ...
                       'gui_OpeningFcn', @instanceSimilarity_OpeningFcn, ...
                       'gui_OutputFcn',  @instanceSimilarity_OutputFcn, ...
                       'gui_LayoutFcn',  [] , ...
                       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end

    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
    % End initialization code - DO NOT EDIT
end

% --- Executes just before instanceSimilarity is made visible.
function instanceSimilarity_OpeningFcn(hObject, eventdata, handles, varargin)
    clc; warning off;
    allAxesInFigure = findall(gcf,'type','axes');
    set(allAxesInFigure,'XTickLabel','','YTickLabel','');
    
    %% Input data
    addpath(genpath('..\..\vlfeat-0.9.18\toolbox'));   
    expDir = '..\..\vlfeat-0.9.18\apps\recognition';
    expr = 'ex-dbLite16_100-fv-aug';
    type = 'saliency'; %'manual';
    
    global coord images imgDir testInst testInd makeList run curRun traintestData startPos trainInst trainInstInd trainInd
    
    load(fullfile('data-all',['fullData-full-' type '.mat']));
    load(fullfile(expDir,'data',expr,'imdb.mat'));
    load(['Results' filesep 'Run-saliency-all_inst_MI_SVM -Kernel 0.mat']);
    load([expDir filesep imageDir filesep 'regionCoordsSaliency-org.mat']);
    
    makeList = unique({traintestData{:,3}});
    testInd = find(images.set==3);
    trainInd = find(images.set==1);
    imgDir = [expDir filesep imageDir];
    % Define testset instances
    startPos = find(strcmp('inst-1', {traintestData{:,1}}));
    testInst = []; trainInst = []; trainInstInd = [];
    for index=1:length(testInd)
        curfilename = regexp(images.name(testInd(index)),'\','split');
        correcttestInd = find(ismember({traintestData{:,2}},curfilename{1}(2)));
        testInst = [testInst; ones(length(correcttestInd),1).*index];
    end
    for index=1:length(trainInd)
        curfilename = regexp(images.name(trainInd(index)),'\','split');
        correcttrainInd = find(ismember({traintestData{:,2}},curfilename{1}(2)));
        trainInst = [trainInst; ones(length(correcttrainInd),1).*index];
        trainInstInd = [trainInstInd; correcttrainInd'];
    end
    
    set(handles.cmbMake,'string',makeList);
    curClass = get(handles.cmbMake, 'Value');
    curRun = run{curClass};
    loadImages(hObject, handles);
   
    handles.output = hObject;

    % Update handles structure
    guidata(hObject, handles);
end

% --- Outputs from this function are returned to the command line.
function varargout = instanceSimilarity_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;
end

% --- Executes during object creation, after setting all properties.
function cmbMake_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

% --- Executes during object creation, after setting all properties.
function cmbImage_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

% --- Executes during object creation, after setting all properties.
function cmbInstance_CreateFcn(hObject, eventdata, handles)
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
end

% --- Executes on button press in btnDisplay.
function btnDisplay_Callback(hObject, eventdata, handles)
    global startPos traintestData curtestInd trainInstInd
    curBag = get(handles.cmbImage, 'Value');
    curtestInstInd = startPos(curtestInd(curBag))+get(handles.cmbInstance, 'Value')-1;
    curFeature = traintestData{curtestInstInd,end};
    featDist = pdist2(curFeature,cell2mat({traintestData{trainInstInd,end}}'),'cosine');
    [distVal, distInd] = sort(featDist);
    displayMatches(handles, distVal, distInd);
end

function displayMatches(handles, distVal, distInd)
    global trainInst trainInd imgDir images coord traintestData trainInstInd
    
    for index = 1:10
        imageInd = trainInd(trainInst(distInd(index)));
        instIndtmp = find(trainInst == trainInst(distInd(index)));
        instInd = find(instIndtmp==distInd(index));
        curImg = imread([imgDir filesep images.name{imageInd}]); 
        axes(eval(['handles.axes' num2str(index)])); imshow(curImg); hold on
        rectangle('Position',coord(imageInd).bb(instInd,:),'EdgeColor','g','lineWidth',1.5);
        hold off
        set(eval(['handles.lbl' num2str(index)]),'String',num2str(distVal(index)));
        set(eval(['handles.txt' num2str(index)]),'String', [traintestData{trainInstInd(distInd(index)),3} ' ']);
    end
end

% --- Executes on selection change in cmbMake.
function cmbMake_Callback(hObject, eventdata, handles)
    global run curRun
    curClass = get(handles.cmbMake, 'Value');
    curRun = run{curClass};
    set(handles.cmbInstance,'value',1);   
    loadImages(hObject, handles);
    resetHandles(handles);
end

% --- Executes on selection change in cmbImage.
function cmbImage_Callback(hObject, eventdata, handles)
    set(handles.cmbInstance,'value',1);   
    loadImages(hObject, handles);
    resetHandles(handles);
end

% --- Executes on selection change in cmbInstance.
function cmbInstance_Callback(hObject, eventdata, handles)
    [~] = displayRegions(handles);
    resetHandles(handles);
end

%% Loading test data
function loadImages(hObject, handles)
    global images curtestInd confInst testInd makeList curRun
    curClass = get(handles.cmbMake, 'Value');
    curInd = find(images.class==curClass);
    curtestInd = intersect(testInd,curInd);
    filename = cellfun(@(x) regexp(x,'\','split'), {images.name{curtestInd}},'uni',0);
    curList = cellfun(@(x)(x{2}), filename, 'uni', 0);
    set(handles.cmbImage,'string',curList);
    
    load(['data-all' filesep 'data_' makeList{curClass} '_test_0_nans.mat']);
    deletedInds  = find(nanInd~=0);
    newInstLb = curRun.inst_label;
    confInst = curRun.inst_prob;
    for ii=1:length(deletedInds)
        newInstLb = [newInstLb(1:deletedInds(ii)-1);0;newInstLb(deletedInds(ii):end)];
        confInst = [confInst(1:deletedInds(ii)-1);0;confInst(deletedInds(ii):end)];
    end
       
    instList = displayRegions(handles);
    set(handles.cmbInstance,'string',instList);     
end

function [instList] = displayRegions(handles)
    global coord images imgDir testInst testInd
    global curtestInd confInst
    
    curBag = get(handles.cmbImage, 'Value');
    instInd = find(testInst==find(testInd ==curtestInd(curBag)));
    instList = {};
    curImg = imread([imgDir filesep images.name{curtestInd(curBag)}]); 
    coordInd = find(strcmp({coord.filename}, images.name{curtestInd(curBag)})==1);
    axes(handles.axesImg); imshow(curImg); hold on
    for index=1:length(instInd)
        instList{index} = ['inst-' num2str(index)];
        rectangle('Position',coord(coordInd).bb(index,:),'EdgeColor','y','lineWidth',1.0);
        if index==get(handles.cmbInstance, 'Value')
            rectangle('Position',coord(coordInd).bb(index,:),'EdgeColor','r','lineWidth',1.5);
            text(coord(coordInd).bb(index,1),coord(coordInd).bb(index,2),num2str(confInst(instInd(index))),...
                 'BackGroundColor','w','FontWeight','b','fontSize',7.5);
        end
    end
    hold off
end

%% Clear handles
function resetHandles(handles)
    % Clear results
    for i = 1:10
        axes(eval(strcat('handles.',['axes' num2str(i)]))); cla();
        set(gca,'XTickLabel','','YTickLabel','');
        set(eval(['handles.lbl' num2str(i)]),'String','');
        set(eval(['handles.txt' num2str(i)]),'String','');
    end
end
