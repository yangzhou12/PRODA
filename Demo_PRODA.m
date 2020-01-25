% Load dataset
load FERETC80A45; % Each sample is a 32x32 matrix
[dc, dr, numSpl] = size(fea2D); % 32x32x320

% Partition the training and test sets
load DBpart; % 2 images per class for training, 2 image per class for test
fea2D_Train = fea2D(:, :, trainIdx);
gnd_Train = gnd(trainIdx);

Py = 500; Pz = 500; Iter = 500;
regParam = 1e2; % Could be tuned for differnt applications
[ model ] = PRODA( fea2D_Train, gnd_Train, ...
    'Py', Py, 'Pz', Pz, 'regParam', regParam, ...
    'maxIter', Iter, 'tol', 1e-4); 

% PRODA Projection
newfea = projPRODA(fea2D, model);

% Sort the projected features by Fisher scores
[odrIdx, stFR] = sortProj(newfea(:,trainIdx), gnd(trainIdx));
newfea = newfea(odrIdx,:); 

% Classification via 1NN
dimTest = 200; % the number of features to be fed into a classifier
testfea = newfea(1:dimTest,:);
% In practice, it would be better to test different values of dimTest 
% for the best classification performance.

nnMd = fitcknn(testfea(:,trainIdx)', gnd(trainIdx));
label = predict(nnMd, testfea(:,testIdx)');
Acc = sum(gnd(testIdx) == label)/length(testIdx)
