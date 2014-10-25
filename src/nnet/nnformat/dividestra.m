function [trainV,valV,testV,trainInd,valInd,testInd] = dividestra(allV,trainRatio,valRatio,testRatio,targets)
%DIVIDESTRA Divide vectors into three sets using stratified indices.
%
% Syntax
%
%   [trainV,valV,testV,trainInd,valInd,testInd] =
%     dividestra(allV,trainRatio,valRatio,testRatio)
%
% Description
%
%   DIVIDESTRA is used to separate input and target vectors into three
%   sets: training, validation and testing.
% 
%   DIVIDESTRA takes the following inputs,
%     allV       - RxQ matrix of Q R-element vectors.
%     trainRatio - Ratio of vectors for training, default = 0.8.
%     valRatio   - Ratio of vectors for validation, default = 0.2.
%     testRatio  - Ratio of vectors for testing, default = 0.
%   and returns:
%     trainV   - Training vectors
%     valV     - Validation vectors
%     testV    - Test vectors
%     trainInd - Training indices
%     valInd   - Validation indices
%     testInd  - Test indices
%
% Examples
%
%     p = rands(3,1000);
%     t = [p(1,:).*p(2,:); p(2,:).*p(3,:)];
%     [trainP,valP,testV,trainInd,valInd,testInd] = dividestra(p,0.8,0.2,0);
%     [trainT,valT,testT] = divideind(t,trainInd,valInd,testInd);
%
%  Network Use
%
%   Here are the network properties that defines which data division function
%   to use, and what its parameters are, when TRAIN is called.
%
%     net.divideFcn
%     net.divideParam
%
% See also divideblock, divideind, divideint, dividerand.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

%% FUNCTION INFO
if ischar(allV)
  switch (allV)
    case 'info'
      info.name = mfilename;
      info.title = 'Stratified';
      info.type = 'Data Division';
      info.version = 6;
      trainV = info;
    case 'name'
      trainV = 'Stratified';
		case 'fpdefaults'
      defaults = struct;
      defaults.trainRatio = 0.8;
      defaults.valRatio = 0.2;
      defaults.testRatio = 0;
			defaults.targets = [];
      trainV = defaults;
    otherwise
      error('NNET:Arguments','Unrecognized string: %s',allV)
  end
  return
end

%% ERROR CHECKING AND DEFAULTS
if isstruct(trainRatio)
  valRatio = trainRatio.valRatio;
  testRatio = trainRatio.testRatio;
	targets = trainRatio.targets;
  trainRatio = trainRatio.trainRatio;
elseif (nargin < 4)
	error('NNET:Arguments','Not enough arguments.')
end

%% DIVIDE DATA
[allV,mode] = nnpackdata(allV);

% stratified divide
% [trainInd,valInd] = crossvalind('HoldOut',targets,trainRatio,'classes','1');
% for i=2:max(targets)
% 	[tra,val] = crossvalind('HoldOut',targets,trainRatio,'classes',num2str(i));
% 	
% 	% apply logic OR
% 	trainInd = trainInd|tra;
% 	valInd = valInd|val;
% end
[trainInd,valInd] = crossvalind('HoldOut',targets,1-trainRatio);
testInd = [];

trainV = allV{1,1}(trainInd);
valV = allV{1,1}(valInd);
testV = [];
