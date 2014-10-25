function datasetNew = transdata(dataset)
%TRANSDATA Transform initial targets dataset in a new targets dataset.
%
%  Syntax
%
%    datasetNew = transdata(dataset)
%
%  Description
%
%    TRANSDATA(dataset) takes,
%      dataset - Targets dataset.
%
%  Examples
%
%    % targets vector
%    targets = [1 1 0 0 0 0; 0 0 1 1 0 0; 0 0 0 0 1 1];
% 
%    % transform initial targets
%	   TargetsNew = transdata(targets);
%
%  See also CONVDATA, CONVOUTPUTS.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

%% ERROR CHECKING
if (nargin < 1), error('NNET:Arguments','Not enough arguments.'),end

%% DATA TRANSFORMATION
datasetNew = dataset';

for j=1:size(datasetNew,2)-1
	for i=1:size(datasetNew,1)
		if datasetNew(i,j) == 1
			datasetNew(i,j+1:size(datasetNew,2)) = ones(size(datasetNew(i,j+1:size(datasetNew,2))));
		end
	end
end

%% return value
datasetNew = datasetNew';
