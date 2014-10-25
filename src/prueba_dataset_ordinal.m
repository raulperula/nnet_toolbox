%%% uso del conjunto de datos balance en redes neuronales

% load inputs and targets of a dataset
loaddata;

%% using train dataset

% creating the neural network
net = newoff(trainInputs,trainTargets,20,'logsig');
net.trainParam.showWindow = false; % don't show training interface

% training the net
net = otrain(net,trainInputs,trainTargetsNew);

%% using test dataset

% simulating the net to obtain the outputs
testOutputs = osim(net,testInputs);

%% medidas de error

% calculating and ploting confussion matrix
[c,cm,ind,per] = confusion(testTargets,testOutputs);
saveas(plotconfusion(testTargets,testOutputs),'mc.png');

% calculating CCR and MAE
ccrcalc(cm, size(testInputs,2))
maecalc(cm, size(testInputs,2))
