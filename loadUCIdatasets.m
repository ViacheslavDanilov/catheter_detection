% Vaishali R | VIT University   %
% Datasets for Feature Selection%
% Email: vrv.vaishali@gmail.com%
%
%%

clc; clear all;


% 1. Zoo Dataset
zoo=load('MAT files\UCI datasets\zoo.dat');
fprintf('Zoo.dat Loaded Successfully \n');

%2. Wine Dataset
winetemp=load('MAT files\UCI datasets\wine.mat');
wine=winetemp.A;
fprintf('wine.mat Loaded Successfully \n');

% 3. Votes Dataset

votestemp=load('MAT files\UCI datasets\votes.mat');
votes=votestemp.votes;
votes=knnimpute(votes); % KNN impute handles missing data in the votes dataset
fprintf('votes.mat Loaded Successfully \n');

% 4. SPECT dataset

specttemp=load('MAT files\UCI datasets\spect.mat');
spect=specttemp.output;
fprintf('spect.mat Loaded Successfully \n');

% 5. Semeion Dataset
semeion=load('MAT files\UCI datasets\semeion.dat');
fprintf('semeion.dat Loaded Successfully \n');

% 6. ILPD dataset
ilpdtemp=load('MAT files\UCI datasets\lipid.mat');
ilpd=ilpdtemp.ilpd;
fprintf('ilpd Loaded Successfully \n');

% 7. isolet5 Dataset

isolet=load('MAT files\UCI datasets\isolet5.dat');

fprintf('isolet5.dat Loaded Successfully \n');


% 8. Ionosphere Dataset

ionosphere=load('MAT files\UCI datasets\ionosphere.csv');
fprintf('ionosphere.csv Loaded Successfully \n');

% 9. Heart Disease dataset
heart=load('MAT files\UCI datasets\heart.dat');
fprintf('heart.dat Loaded Successfully \n');

% 10. Glass Dataset
glass= load('MAT files\UCI datasets\glass.dat');
fprintf('glass.dat Loaded Successfully \n');

% 11. COIL20 Dataset
coiltemp=load('MAT files\UCI datasets\coil.mat');
coil=coiltemp.b;
fprintf('coil.mat Loaded Successfully \n');

% 12. Clean1 Dataset (MUSK)
clean1= load('MAT files\UCI datasets\clean1.csv');
fprintf('clean1.csv Loaded Successfully \n');

% 13. BreastEW Dataset
btemp=load('MAT files\UCI datasets\breastEW.mat');
breastEW=btemp.data;
fprintf('breastEW.mat Loaded Successfully \n');


fprintf('All datasets are loaded Successfully!\n');
