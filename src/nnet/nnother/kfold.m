function [NHO,E] = kfold(trainInputs,trainTargets,NH,kFold,numIter)
%KFOLD Do k-Fold cross-validation.
%
%  Syntax
%
%    kfold(trainInputs,trainTargets,NHmax,kFold,numIter)
%
%  Description
%
%    KFOLD(trainInputs,trainTargets,NHmax,kFold,numIter) takes,
%      trainInputs  - Train inputs of a dataset.
%      trainTargets - Train targets of a dataset.
%      NH           - Vector of hidden nodes.
%      kFold        - Number of folds.
%      numIter      - Mean number of calculations.
%    and returns:
%	   NHO          - Number of optimal hidden nodes.
%      E            - Mean error.
%
%  Examples
% 
%
%  See also transdata, newoff, simonet, maecalc.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $


%% ERROR CHECKING
if (nargin < 2), error('NNET:Arguments','Not enough arguments.'),end

%% DEFAULTS
if (nargin < 3), NH = 1:5:51; end
if (nargin < 4), kFold = 10; end
if (nargin < 5), numIter = 3; end

%% establecimiento de las clases
classes = getstra(trainTargets);

%% K-FOLD CALCULATING

% geting indices from stratified set of targets
indices = crossvalind('Kfold',classes,kFold);

v = zeros(length(NH),1);

for i=NH
	auxmae = 0;
	for k=1:kFold
		% calculating and dividing folds
		testing = (indices == k)';
		training = ~testing;
        
		% Training set
		trainX = trainInputs(:,training);
		trainY = trainTargets(:,training);

		% Test set
		testX = trainInputs(:,testing);
		testY = trainTargets(:,testing);
	
		for j=1:numIter
			% creating the neural network
			net = newff(trainX,trainY,i,{'tansig','logsig'},'trainirp','learngdm','mse',{'fixunknowns','removeconstantrows','mapminmax'},{},'dividestra');
			net.trainParam.showWindow = false; % don't show training interface

			% adjust dataset divide
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.divideParam.targets = getstra(trainY);

			% training the net
			net = train(net,trainX,trainY);

			% simulating the net to obtain the outputs (test set)
			testOutputs = sim(net,testX);
			
			% tratamiento de los valores de salida
			aux = zeros(size(testOutputs));
			[c,index] = max(testOutputs);
			l = 1;
			for h=index
				aux(h,l) = 1;
				l = l+1;
			end
			testOutputs = aux;

			% calculating confussion matrix and mae
			[c,cm,ind,per] = confusion(testY,testOutputs);
			auxmae = auxmae+maecalc(cm, size(testX,2));
		end
	end
	% obtaining mean error
	v(NH == i) = auxmae/(kFold*numIter);
end

%% return values
NHO = NH(find(v == min(v),1,'first'));
E = min(v);
