function net = newoff(varargin)
%NEWFFO Create an ordinal feed-forward backpropagation network.
%
%  Syntax
%
%    net = newffo(P,T,S,TF)
%
%  Description
%
%    NEWOFF(P,T,S,TF) takes,
%      P  - RxQ1 matrix of Q1 representative R-element input vectors.
%      T  - SNxQ2 matrix of Q2 representative SN-element target vectors.
%      Si  - Sizes of N-1 hidden layers, S1 to S(N-1), default = [20 1].
%            (Output layer size SN is determined from T.)
%      TFi - Transfer function of ith layer. Default is 'logsig' for
%            hidden layers, 'purelin' for the last hidden layer and 'logsig' for output layer.
%    and returns an N layer ordinal feed-forward backprop network.
%
%    The transfer functions TF{i} can be any differentiable transfer
%    function such as TANSIG, LOGSIG, or PURELIN.
%
%    *WARNING*: TRAINIRPO is the default training function because it
%    is very fast, but it requires a lot of memory to run.
%
%    The learning function can be either of the backpropagation
%    learning functions such as LEARNGD, or LEARNGDM.
%
%    The performance function can be any of the differentiable performance
%    functions such as MSE or MSEREG.
%
%  Examples
%
%    load iris_dataset
%    net = newoff(irisInputs,irisTargets,20,logsig);
%    net = train(net,irisInputs,irisTargets);
%    irisOutputs = osim(net,irisInputs);
%
%  Algorithm
%
%    Ordinal feed-forward networks consist of Nl layers using the DOTPROD
%    weight function, NETSUM net input function, and the specified
%    transfer functions.
%
%    The first layer has weights coming from the input.  Each subsequent
%    layer has a weight coming from the previous layer.  All layers
%    have biases except the middle layer.  The last layer is the network output.
%
%    Each layer's weights and biases are initialized with INITNW.
%
%    Adaption is done with TRAINS which updates weights with the
%    specified learning function. Training is done with the specified
%    training function. Performance is measured according to the specified
%    performance function.
%
%  See also NEWFF, OSIM, INIT, TRAINIRP, TRAINIRPO.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

%% ERROR CHECKING
if nargin < 2, error('NNET:Arguments','Not enough input arguments'), end

%% CREATE NEURAL NETWORK
if (nargin == 2)
	net = newff(varargin{1},varargin{2}(1:(size(varargin{2},1)-1),:),[20 1],{'tansig','purelin','logsig'},'trainirpo',...
		'learngdm','mse',{'fixunknowns','removeconstantrows','mapminmax'},{},'dividestra');
elseif (nargin == 3)
	net = newff(varargin{1},varargin{2}(1:(size(varargin{2},1)-1),:),[varargin{3} 1],{'tansig','purelin','logsig'},'trainirpo',...
		'learngdm','mse',{'fixunknowns','removeconstantrows','mapminmax'},{},'dividestra');
elseif (nargin == 4)
	if isa(varargin{4},'cell') % transformation if it is a cell array
		aux = varargin{4};
		aux(:,length(aux)+1:length(aux)+2) = {'purelin','logsig'};
		varargin{4} = aux;
		
		net = newff(varargin{1},varargin{2}(1:(size(varargin{2},1)-1),:),[varargin{3} 1],varargin{4},'trainirpo',...
			'learngdm','mse',{'fixunknowns','removeconstantrows','mapminmax'},{},'dividestra');
	else
		net = newff(varargin{1},varargin{2}(1:(size(varargin{2},1)-1),:),[varargin{3} 1],{varargin{4},'purelin','logsig'},'trainirpo',...
			'learngdm','mse',{'fixunknowns','removeconstantrows','mapminmax'},{},'dividestra');
	end
else
	error('NNET:Arguments','Input arguments incorrect');
end

%% ADJUST PARAMETERS

% set the name of de ANN
net.name = 'Ordinal Neural Network';
% delete bias of the last hidden layer
net.biasConnect(net.numLayers-1) = 0;
% put weight of output layer to one fixed
net.LW{net.numLayers,net.numLayers-1} = ones(net.outputs{net.numLayers}.size,1);
% initialize with correct bias values
net.b{net.numLayers} = (-5*((size(net.b{net.numLayers},1)-1)/2):5:5*((size(net.b{net.numLayers},1)-1)/2))';

% adjust dataset divide
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;
net.divideParam.targets = getstra(varargin{2});
