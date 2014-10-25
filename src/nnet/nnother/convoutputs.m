function outputs = convoutputs(outputsOld)
%CONVOUTPUTS Transform old outputs dataset in a new outputs dataset.
%
%  Syntax
%
%    outputs = convoutputs(outputsOld)
%
%  Description
%
%    CONVOUTPUTS(outputsOld) takes,
%      outputsOld - Old outputs dataset.
%
%  Examples
%
%    % outputs vector
%    outputs = [0.50071,0.60006,0.30003,0.4,0.5; 0.91916,0.90177,0.80103,0.90001,0.10024; 1,1,1,1,1;];
% 
%    % transform old outputs
%	   Outputs = convoutputs(outputs);
%
%  See also CONVDATA, TRANSDATA.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

%% ERROR CHECKING
if (nargin < 1), error('NNET:Arguments','Not enough arguments.'),end

%% obtain outputs dataset of no accumulated probabilities

%%% iterative mode (no valid)
% aux = zeros(size(outputsOld,1),1);
% for i=1:size(outputsOld,2)
% 	aux(1) = outputsOld(1,i);
% 	for j=2:size(outputsOld,1)
% 		aux(j) = outputsOld(j,i)-outputsOld(j-1,i);
% 	end
% 	outputsOld(:,i) = aux;
% end

%%% matricial mode
outputsaux = circshift((circshift(outputsOld,-1).*[ones(size(outputsOld,1)-1,size(outputsOld,2)); zeros(1,size(outputsOld,2))])-outputsOld,1);
outputsaux(1,:) = outputsOld(1,:);

%% boolean transform of outputs dataset
aux = zeros(size(outputsaux));
[c,i] = max(outputsaux);
j = 1;
for k=i
	aux(k,j) = 1;
	j = j+1;
end

%% return value
outputs = aux;
