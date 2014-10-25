function CCR = ccrcalc(MC, TOTAL)
%CCRCALC Calculate Correctly Classified Rate.
%
%  Syntax
%
%    CCR = ccrcalc(MC, TOTAL)
%
%  Description
%
%    CCRCALC(MC, TOTAL) takes,
%      MC  - Confusion matrix.
%      TOTAL - Total clases elements.
%
%  Examples
%
%  % targets and outputs vectors
%  targets = [1 1 0 0 0 0; 0 0 1 1 0 0; 0 0 0 0 1 1];
%  outputs = [0 1 0 0 1 0; 0 0 0 1 1 0; 0 0 0 0 1 1];
% 
%	 % confussion matrix calc
%  [c,cm,ind,per] = confusion(targets,outputs);
% 
%  % CCR calc
%  ccrcalc(cm, size(targets,2))
%
%  See also TRACE, MAE, MSE, MAECALC, CONFUSION, PLOTCONFUSION.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

% CCR calculation
CCR = trace(MC)/TOTAL;
