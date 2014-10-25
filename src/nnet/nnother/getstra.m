function classes = getstra(trainTargets)
%GETSTRA Get stratifiedclasses.
%
%  Syntax
%
%    getstra(trainTargets)
%
%  Description
%
%    GETSTRA(trainTargets) takes,
%      trainTargets - Train targets of a dataset.
%    and returns:
%		   classes      - Vector with stratified classes.
%
%  Examples
% 
%
%  See also kfold.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $


%% ERROR CHECKING
if (nargin < 1), error('NNET:Arguments','Not enough arguments.'),end

classes = zeros(1,max(size(trainTargets)));
for i=1:min(size(trainTargets))
	classes = classes+((trainTargets(i,:))*i);
end
