function MAE = maecalc(MC, TOTAL)
%MAECALC Calculate Mean Absolute Error.
%
%  Syntax
%
%    MAE = maecalc(A)
%
%  Description
%
%    MAECALC(MC, TOTAL) takes,
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
%  % MAE calc
%  maecalc(cm, size(targets,2))
%
%  See also TRACE, CCRCALC, MAE, MSE, CONFUSION, PLOTCONFUSION.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

% wieght matrix calculation
for i=1:size(MC, 1)
	aux = i-1;
	for j=1:size(MC, 2)
		W(i,j) = abs(aux);
		aux = aux-1;
	end
end

% multiplication of confusion matrix by weight matrix
MCnew = MC.*W;

% MAE calculation
MAE = sum(sum(MCnew))/TOTAL;
