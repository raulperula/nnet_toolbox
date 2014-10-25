function [net,tr] = otrain(net,X,T)
%OTRAIN Train an ordinal neural network.
%
%  Syntax
%
%    [net,tr] = otrain(NET,X,T)
%
%  Description
%
%    OTRAIN trains an ordinal network NET according to NET.trainFcn and
%    NET.trainParam.
%
%    OTRAIN(NET,X,T) takes,
%      NET - Network.
%      X   - Network inputs.
%      T   - Network targets.
%    and returns,
%      NET - New network.
%      TR  - Training record (epoch and perf).
%
%  See also train.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $


%% ARGUMENT CHECKS
if nargin < 3, error('NNET:Arguments','Not enough input arguments.'); end

%% TRAIN FUNCTION CALLED
[net,tr] = train(net,X,T(1:(size(T,1)-1),:));
