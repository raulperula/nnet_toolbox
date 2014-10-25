function convdata(name, data, C, O, S)
%CONVDATA Create a valid matlab inputs and targets datasets with a conversion of a invalid dataset.
%
%  Syntax
%
%    convdata(name, data, C, O, S)
%
%  Description
%
%    CONVDATA(name, data, C, O, S) takes,
%      name  - Name of dataset.
%      data  - Invalid dataset.
%      C     - Number of classes.
%      O     - Number of outputs.
%      S     - Save the inputs and targets in a mat file.
%
%  Examples
% 
%    % initialize values
%    name = 'train';
%    C = 4;
%    O = 3;
%    data = [2,3,2,4,1,0,0; 4,1,2,5,1,0,0; 1,1,2,2,0,1,0; 2,1,5,5,0,1,0; 1,1,4,1,0,0,1;];
% 
%    % convert invalid data in valid datasets
%    convdata(name, data, C, O);
%
%  See also TRANSDATA, CONVOUTPUTS.

% Raúl Pérula Martínez, 07-2011
% Copyright 2011 Universidad de Córdoba
% $Revision: 1.0 $

%% ERROR CHECKING
if nargin < 4, error('NNET:Arguments','Not enough arguments.'),end

%% DATA CONVERSION

% data matrix transposition
data = data';

% creating variable with inputs
str = [name 'Inputs'];
var = genvarname(str);
assignin('base',var,data(1:C,:));

if (nargin == 5 && S == true), save([name '_dataset.mat'], var); end

% creating variable with targets
str = [name 'Targets'];
var = genvarname(str);
assignin('base',var,data(C+1:C+O,:));

if (nargin == 5 && S == true), save([name '_dataset.mat'], var, '-append'); end
