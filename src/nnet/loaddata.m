
% get elements in the directory
d = dir('datasets/10-holdout/ERA_nnep');

for i=[3 13]
	% import file
	importfile(d(i).name);

	% get parameters
	v = sscanf(textdata{1}, '%d %d %d');

	% put params correctly
	name = strtok(d(i).name,'_');
	C = v(2);
	O = v(3);

	% converting data
	convdata(name, data, C, O);
end

% transform original train targets to final train targets
trainTargetsNew = transdata(trainTargets);

% clearing the innecesaries variables
clear d colheaders textdata data v name C O i
