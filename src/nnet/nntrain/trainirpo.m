function [net,tr] = trainirpo(net,tr,trainV,valV,testV,varargin)
%TRAINRP iRPROP+ ordinal backpropagation.
%
%  Syntax
%  
%    [net,tr,Ac,El] = trainirpo(net,tr,trainV,valV,testV)
%    info = trainirpo('info')
%
%  Description
%
%    TRAINIRPO is a network training function that updates weight and
%    bias values according to the resilient backpropagation algorithm
%     (iRPROP+).
%
%    TRAINIRPO(NET,TR,TRAINV,VALV,TESTV) takes these inputs,
%      NET - Neural network.
%      TR  - Initial training record created by TRAIN.
%      TRAINV - Training data created by TRAIN.
%      VALV - Validation data created by TRAIN.
%      TESTV - Test data created by TRAIN.
%    and returns,
%      NET - Trained network.
%      TR  - Training record of various values over each epoch.
%
%    Each argument TRAINV, VALV and TESTV is a structure of these fields:
%      X  - NxTS cell array of inputs for N inputs and TS timesteps.
%           X{i,ts} is an RixQ matrix for ith input and ts timestep.
%      Xi - NxNid cell array of input delay states for N inputs and Nid delays.
%           Xi{i,j} is an RixQ matrix for ith input and jth state.
%      Pd - NxSxNid cell array of delayed input states.
%      T  - NoxTS cell array of targets for No outputs and TS timesteps.
%           T{i,ts} is an SixQ matrix for the ith output and ts timestep.
%      Tl - NlxTS cell array of targets for Nl layers and TS timesteps.
%           Tl{i,ts} is an SixQ matrix for the ith layer and ts timestep.
%      Ai - NlxTS cell array of layer delays states for Nl layers, TS timesteps.
%           Ai{i,j} is an SixQ matrix of delayed outputs for layer i, delay j.
%
%    Training occurs according to training parameters, with default values:
%      net.trainParam.show        25  Epochs between displays
%      net.trainParam.showCommandLine 0 generate command line output
%      net.trainParam.showWindow   1 show training GUI
%      net.trainParam.epochs     100  Maximum number of epochs to train
%      net.trainParam.goal         0  Performance goal
%      net.trainParam.time       inf  Maximum time to train in seconds
%      net.trainParam.min_grad  1e-6  Minimum performance gradient
%      net.trainParam.max_fail     5  Maximum validation failures
%      net.trainParam.lr        0.01  Learning rate
%      net.trainParam.delt_inc   1.2  Increment to weight change
%      net.trainParam.delt_dec   0.5  Decrement to weight change
%      net.trainParam.delta0    0.07  Initial weight change
%      net.trainParam.deltamax  50.0  Maximum weight change
%
%    TRAINIRPO('info') returns useful information about this function.
%
%  Network Use
%
%    You can create a standard network that uses TRAINIRPO with
%    NEWFF, NEWCF, or NEWELM.
%
%    To prepare a custom network to be trained with TRAINIRPO:
%    1) Set NET.trainFcn to 'trainirpo'.
%       This will set NET.trainParam to TRAINIRPO's default parameters.
%    2) Set NET.trainParam properties to desired values.
%
%    In either case, calling TRAIN with the resulting network will
%    train the network with TRAINIRPO.
%
%  Examples
%
%    Here is a problem consisting of inputs P and targets T that we would
%    like to solve with a network.
%
%      p = [0 1 2 3 4 5];
%      t = [0 0 0 1 1 1];
%
%    Here a two-layer feed-forward network is created.  The network's
%    input ranges from [0 to 10].  The first layer has two TANSIG
%    neurons, and the second layer has one LOGSIG neuron.  The TRAINIRPO
%    network training function is to be used.
%
%      % Create and Test a Network
%      net = newoff(p,t,20,'logsig','trainirpo');
%      a = sim(net,p)
%
%      % Train and Retest the Network
%      net.trainParam.epochs = 50;
%      net.trainParam.show = 10;
%      net.trainParam.goal = 0.1;
%      net = train(net,p,t);
%      a = sim(net,p)
%
%    See NEWOFF, NEWFF, and TRAINIRP for other examples.
%
%  Algorithm
%
%    TRAINIRPO can train any network as long as its weight, net input,
%    and transfer functions have derivative functions.
%
%    Ordinal backpropagation is used to calculate derivatives of performance
%    PERF with respect to the weight and bias variables X.  Each
%    variable is adjusted according to the following:
%
%      dX = deltaX.*sign(gX);
%
%     where the elements of deltaX are all initialized to delta0 and
%     gX is the gradient.  At each iteration the elements of deltaX
%     are modified.  If an element of gX changes sign from one 
%     iteration to the next, then the corresponding element of
%     deltaX is decreased by delta_dec.  If an element of gX 
%     maintains the same sign from one iteration to the next,
%     then the corresponding element of deltaX is increased by
%     delta_inc.  See Reidmiller, Proceedings of the IEEE Int. Conf. 
%      on NN (ICNN) San Francisco, 1993, pp. 586-591.
%
%    Training stops when any of these conditions occur:
%    1) The maximum number of EPOCHS (repetitions) is reached.
%    2) The maximum amount of TIME has been exceeded.
%    3) Performance has been minimized to the GOAL.
%    4) The performance gradient falls below MINGRAD.
%    5) Validation performance has increased more than MAX_FAIL times
%       since the last time it decreased (when using validation).
%
%  See also NEWOFF, NEWFF, TRAINRP, TRAINIRP.
%
%   References
%
%     Christian Igel, Michael Hüsken, Empirical evaluation of the improved Rprop learning algorithms
%     Institut für Neuroinformatik, Ruhr-Universitat Bochum, 44780 Bochum, Germany

% Updated by Raúl Pérula Martínez
% Copyright 2011
% $Revision: 1.0 $

%% Info
if strcmp(net,'info')
  info.function = mfilename;
  info.title = 'iRProp';
  info.type = 'Training';
  info.version = 6;
  info.training_mode = 'Supervised';
  info.gradient_mode = 'Gradient';
  info.uses_validation = true;
  info.param_defaults.show = 25;
  info.param_defaults.showWindow = true;
  info.param_defaults.showCommandLine = false;
  info.param_defaults.epochs = 1000;
  info.param_defaults.time = inf;
  info.param_defaults.goal = 0;
  info.param_defaults.max_fail = 6;
  info.param_defaults.min_grad = 1e-10;
  info.param_defaults.delt_inc = 1.2;
  info.param_defaults.delt_dec = 0.5;
  info.param_defaults.delta0 = 0.07;
  info.param_defaults.deltamax = 50.0;
    
  info.training_states = ...
    [ ...
    training_state_info('gradient','Gradient','continuous','log') ...
    training_state_info('mu','Mu','continuous','log') ...
    training_state_info('val_fail','Validation Checks','discrete','linear') ...
    ];
  net = info;
  return
end

%% NNET 5.1 Backward Compatibility
if ischar(net)
  switch (net)
    case 'name', info = feval(mfilename,'info'); net = info.title;
    case 'pnames', info = feval(mfilename,'info'); net = fieldnames(info.param_defaults);
    case 'pdefaults', info = feval(mfilename,'info'); net = info.param_defaults;
    case 'gdefaults', if (tr==0), 'calcgrad'; else net='calcgbtt'; end
    otherwise, error('NNET:Arguments','Unrecognized code.')
  end
  return
end

%% Parameters
epochs = net.trainParam.epochs;
show = net.trainParam.show;
goal = net.trainParam.goal;
time = net.trainParam.time;
min_grad = net.trainParam.min_grad;
max_fail = net.trainParam.max_fail;
delt_inc = net.trainParam.delt_inc;
delt_dec = net.trainParam.delt_dec;
delta0 = net.trainParam.delta0;
deltamax = net.trainParam.deltamax;
gradientFcn = net.gradientFcn;

% Parameter Checking
if (~isa(epochs,'double')) || (~isreal(epochs)) || (any(size(epochs)) ~= 1) || ...
  (epochs < 1) || (round(epochs) ~= epochs)
  error('NNET:Arguments','Epochs is not a positive integer.')
end
if (~isa(show,'double')) || (~isreal(show)) || (any(size(show)) ~= 1) || ...
  (isfinite(show) && ((show < 1) || (round(show) ~= show)))
  error('NNET:Arguments','Show is not ''NaN'' or a positive integer.')
end
if (~isa(goal,'double')) || (~isreal(goal)) || (any(size(goal)) ~= 1) || ...
  (goal < 0)
  error('NNET:Arguments','Goal is not zero or a positive real value.')
end
if (~isa(time,'double')) || (~isreal(time)) || (any(size(time)) ~= 1) || ...
  (time < 0)
  error('NNET:Arguments','Time is not zero or a positive real value.')
end
if (~isa(min_grad,'double')) || (~isreal(min_grad)) || (any(size(min_grad)) ~= 1) || ...
  (min_grad < 0)
  error('NNET:Arguments','Min_grad is not zero or a positive real value.')
end
if (~isa(max_fail,'double')) || (~isreal(max_fail)) || (any(size(max_fail)) ~= 1) || ...
  (max_fail < 1) || (round(max_fail) ~= max_fail)
  error('NNET:Arguments','Max_fail is not a positive integer.')
end
if (~isa(delt_inc,'double')) || (~isreal(delt_inc)) || (any(size(delt_inc)) ~= 1) || ...
  (delt_inc < 1)
  error('NNET:Arguments','Delt_inc is not a real value greater than 1.')
end
if (~isa(delt_dec,'double')) || (~isreal(delt_dec)) || (any(size(delt_dec)) ~= 1) || ...
  (delt_dec < 0) || (delt_dec > 1)
  error('NNET:Arguments','Delt_dec is not a real value between 0 and 1.')
end
if (~isa(delta0,'double')) || (~isreal(delta0)) || (any(size(delta0)) ~= 1) || ...
  (delta0 <= 0)
  error('NNET:Arguments','Delta0 is not a positive real value.')
end
if (~isa(deltamax,'double')) || (~isreal(deltamax)) || (any(size(deltamax)) ~= 1) || ...
  (deltamax <= 0)
  error('NNET:Arguments','Deltamax is not a positive real value.')
end

%% Initialize
Q = trainV.Q;
TS = trainV.TS;
vperf = NaN;
tperf = NaN;
val_fail = 0;
startTime = clock;
X = getx(net);
num_X = length(X);

% Initialize Performance
original_net = net;
best_net = net;
doValidation = ~isempty(valV.indices);
doTest = ~isempty(testV.indices);
[perf,El,trainV.Y,Ac,N,Zb,Zi,Zl] = calcperf2(net,X,trainV.Pd,trainV.Tl,trainV.Ai,Q,TS);
best_perf = perf;

if (doValidation)
  [vperf,ignore,valV.Y] = calcperf2(net,X,valV.Pd,valV.Tl,valV.Ai,valV.Q,valV.TS);
  best_vperf = vperf;
end

%% Training Record
tr.best_epoch = 0;
tr.goal = goal;
tr.states = {'epoch','time','perf','vperf','tperf','gradient','val_fail'};

%% Status
status = ...
  [ ...
  training_status('Epoch','iterations','linear','discrete',0,epochs,0), ...
  training_status('Time','seconds','linear','discrete',0,time,0), ...
  training_status('Performance','','log','continuous',best_perf,goal,best_perf) ...
  training_status('Gradient','','log','continuous',1,min_grad,1) ...
  training_status('Validation Checks','','linear','discrete',0,max_fail,0) ...
  ];
nn_train_feedback('start',net,status);

%% Train

%-- Value of delta0 = 0.07
deltaX = delta0*ones(size(X));
%-- Value of deltamax = 50
deltaMAX = deltamax*ones(size(X));
deltaMIN = 0*ones(size(X));
gX = zeros(size(X));
dX = zeros(size(X));

%-- Value of epochs = 1000
for epoch=0:epochs

  % Performance and Gradient
  gX_old = gX;
	%-- Calculate network outputs, signals, and performance
  [perf,El,trainV.Y,Ac,N,Zb,Zi,Zl] = calcperf2(net,X,trainV.Pd,trainV.Tl,trainV.Ai,Q,TS);
  %-- Save old perf to future use to compare
	perf_old = perf;
	%-- gX calculated by dPerf/dx
  [gX,gradient] = calcgx(net,X,trainV.Pd,Zb,Zi,Zl,N,Ac,El,perf,Q,TS);
  if (epoch == 0)
    gX_old = gX;
  end
  
  % Stopping Criteria
  current_time = etime(clock,startTime);
  [userStop,userCancel] = nntraintool('check');
  if userStop, tr.stop = 'User stop.'; net = best_net;
  elseif userCancel, tr.stop = 'User cancel.'; net = original_net;
  elseif (perf <= goal), tr.stop = 'Performance goal met.'; net = best_net;
  elseif (epoch == epochs), tr.stop = 'Maximum epoch reached.'; net = best_net;
  elseif (current_time >= time), tr.stop = 'Maximum time elapsed.'; net = best_net;
  elseif (gradient <= min_grad), tr.stop = 'Minimum gradient reached.'; net = best_net;
  elseif (doValidation) && (val_fail >= max_fail), tr.stop = 'Validation stop.'; net = best_net;
  end
  
  % Training record
  if doTest
    [tperf,ignore,testV.Y] = calcperf2(net,X,testV.Pd,testV.Tl,testV.Ai,testV.Q,testV.TS);
  end
  tr = tr_update(tr,[epoch current_time perf vperf tperf gradient val_fail]);
  
  % Feedback
  nn_train_feedback('update',net,status,tr,{trainV valV testV}, ...
    [epoch,current_time,best_perf,gradient,val_fail]);
  
  % Stop
  if ~isempty(tr.stop), break, end

  %%% APPLY iRPROP+ UPDATE %%%
	ggX = gX.*gX_old;
	
	%-- valor delt_inc = 1.2, delt_dec = 0.5
	%-- deltaX = matriz del tamaño de los pesos con valor delta0
  deltaX = ((ggX>0)*delt_inc + (ggX<0)*delt_dec + (ggX==0)).*deltaX;
  deltaX = (ggX==0).*deltaX + (ggX>0).*min(deltaX,deltaMAX)+ (ggX<0).*max(deltaX,deltaMIN);
  dX = (ggX>0 | ggX==0).*deltaX.*sign(gX) + (ggX<0 & perf>perf_old).*dX;
	ddX = (((ggX>0 | ggX==0).*dX)-((ggX<0 & perf>perf_old).*dX));

  %-- actualizacion de pesos, modificacion para que los pesos de la salida
  %-- siempre valgan uno
	len = net.outputs{net.numLayers}.size;
  X = X+ddX.*[ones(length(X)-2*len,1); zeros(2*len,1)];
	
	%-- actualizacion de bias
	%-- controlar la condicion de beta1 < beta2 < ... < betaN,
	%-- siendo las betas el valor de las bias de la capa de salida, que en este
	%-- caso
	for i=(length(X)-len+1):length(X)
		%-- comprobar que el siguiente elemento es mayor
		if i == length(X)
			if X(i-1) < (X(i)+ddX(i))
				X(i) = X(i)+ddX(i);
			else
				gX(i) = 0;
			end
		else
			if (X(i)+ddX(i)) < X(i+1)
				X(i) = X(i)+ddX(i);
			else
				gX(i) = 0;
			end
		end
	end

  gX = (~(ggX<0)).*gX;
	
	net = setx(net,X);

  % Validation
  if (doValidation)
    [vperf,ignore,valV.Y] = calcperf2(net,X,valV.Pd,valV.Tl,valV.Ai,valV.Q,valV.TS);
    if (vperf < best_vperf)
      best_net = net;
      best_perf = perf;
      best_vperf = vperf;
      tr.best_epoch = epoch+1;
      val_fail = 0;
    elseif (vperf > best_vperf)
      val_fail = val_fail + 1;
    end
  elseif (perf < best_perf)
    best_net = net;
    best_perf = perf;
    tr.best_epoch = epoch+1;
  end
end

%% Finish
tr = tr_clip(tr);
