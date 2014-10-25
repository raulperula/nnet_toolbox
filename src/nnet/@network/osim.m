function testOutputs = simonet(net,testInputs)
%SIMONET Simulate an ordinal neural network.
%
%  Syntax
%
%    simonet(net,test)
%
%  Description
%
%    SIMONET(net,test) takes,
%      net         - Ordinal Neural Network.
%      test        - Test inputs of a dataset.
%    and returns:
%      testOutputs - Test outputs of simulation.
%
%  Examples
% 
%
%  See also convoutputs.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $


%% ERROR CHECKING
if (nargin < 2), error('NNET:Arguments','Not enough arguments.'),end

%% SIMULATION
% simulating the net to obtain the outputs (test set)
testOutputs = sim(net,testInputs);

% adding ones to the last output
testOutputs(size(testOutputs,1)+1,:) = ones(1,size(testOutputs,2));

% converting outputs values
testOutputs = convoutputs(testOutputs);
