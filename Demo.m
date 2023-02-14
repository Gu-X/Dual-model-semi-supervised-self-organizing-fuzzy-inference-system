clear all
clc
close all
%% Supervised Learning from data
load data_demo
Input.TraData=DTra1; %% Labeled Training data
Input.TraLabel=LTra1; %% Ground Truth
Input.ChunkSize=1000; %% Chunk size
Input.GranLevel=9; %% Level of Granularity 
tic
[Input]=DMS3OFIS(Input,'SL'); %% Train DMS3OFIS in the supervised manner
tt1=toc
%% Classifying unlabelled data
Input.TesData=DTes1; %% Unlabelled data
Input.ChunkSize=1000; %% Chunk size
[Output]=DMS3OFIS(Input,'T'); %% Test the trained DMS3OFIS on the unlabelled data
L_pred=Output.EstLabel; %% Predicted class labels
ConfMat=confusionmat(LTes1,L_pred); %% Calculate the confusion matrix based on the true labels and predicted labels
Acc1=sum(sum(ConfMat.*(eye(length(unique(LTes1))))))./length(LTes1) %% Classification accuracy

%% Semi-supervised Learning from data
Input.TraData=DTra1; %% Labeled Training data
Input.TraLabel=LTra1; %% Ground Truth
Input.ULTraData=DTes1; %% Unlabelled data
Input.ChunkSize=1000;  %% Chunk size
Input.GranLevel=9; %% Level of Granularity 
tic
[Input]=DMS3OFIS(Input,'SSL'); %% Train DMS3OFIS in the semi-supervised manner
tt2=toc
%% Classifying unlabelled data
Input.TesData=DTes1; %% Unlabelled data
Input.ChunkSize=1000; %% Chunk size
[Output]=DMS3OFIS(Input,'T'); %% Test the trained DMS3OFIS on the unlabelled data
SSL_pred=Output.EstLabel; %% Predicted class labels
ConfMat=confusionmat(LTes1,SSL_pred); %% Calculate the confusion matrix based on the true labels and predicted labels
Acc2=sum(sum(ConfMat.*(eye(length(unique(LTes1))))))./length(LTes1)  %% Classification accuracy