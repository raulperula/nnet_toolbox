% add to path
clearvars;
addpath(genpath('../rperula'))

% get elements in the directory
str = 'datasets/10-holdout/';
dout = dir(str); % get name of folder

if ~isdir('results/nominal'), mkdir('results/nominal'); end
f = fopen('results/nominal/resultados.txt','a+'); % results text file	
f2 = fopen(strcat('results/nominal/general.csv'),'a+'); % general.csv file
fprintf(f2,'dataset,CCRMean,CCRSD,MAEMean,MAESD,NHMean,NHSD\n');

for h=3:length(dout)
	d = dir(strcat(str,dout(h).name)); % get name of files
	
	dataset_name = strtok(dout(h).name,'_');
	fprintf(f,'%s\n',dataset_name);
	
	cnt = 1; % numIter
	f1 = fopen(strcat('results/nominal/',dataset_name,'.csv'),'w+'); % dataset.csv file
	fprintf(f1,'numIter,CCR,MAE,NH,CompTime\n');
	
	for i=3:length(d)/2+1 % evaluate pair of files, train and test datasets
		t = clock; % get current time
		
		% import file
		importfile(fullfile(str,dout(h).name,d(i).name));

		% get parameters
		v = sscanf(textdata{1}, '%d %d %d');

		% put params correctly
		name = strtok(d(i).name,'_');
		C = v(2);
		O = v(3);

		% converting data
		convdata(name, data, C, O);

		% import file
		if (strcmp(dataset_name,'depresion'))
			importfile(fullfile(str,dout(h).name,d(i+1).name));
		else
			importfile(fullfile(str,dout(h).name,d(i+10).name));
		end

		% get parameters
		v = sscanf(textdata{1}, '%d %d %d');

		% put params correctly
		if (strcmp(dataset_name,'depresion'))
			name = strtok(d(i+1).name,'_');
		else
			name = strtok(d(i+10).name,'_');
		end
		C = v(2);
		O = v(3);

		% converting data
		convdata(name, data, C, O);
		
		% 10-fold
		vNHO(i-2) = kfold(trainInputs,trainTargets,2.^(0:1:5));
		
		for rep=1:3
			%% using train dataset

			% creating the neural network
			net = newff(trainInputs,trainTargets,vNHO(i-2),{'tansig','logsig'},'trainirp','learngdm','mse',{'fixunknowns','removeconstantrows','mapminmax'},{},'dividestra');
			net.trainParam.showWindow = false; % don't show training interface
            
            % adjust dataset divide
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.divideParam.targets = getstra(trainTargets);

			% training the net
			net = train(net,trainInputs,trainTargets);

			%% using test dataset

			% simulating the net to obtain the outputs
			testOutputs = sim(net,testInputs);
			
			% tratamiento de los valores de salida
			aux = zeros(size(testOutputs));
			[c,index] = max(testOutputs);
			j = 1;
			for k=index
				aux(k,j) = 1;
				j = j+1;
			end
			testOutputs = aux;

			%% medidas de error

			% calculating and ploting confussion matrix
			[c,cm,ind,per] = confusion(testTargets,testOutputs);
 			plotname = strcat('results/nominal/',dataset_name,'_mc',num2str(cnt),'.png');
 			saveas(plotconfusion(testTargets,testOutputs),plotname);
 			set(gcf,'visible','off');

			% calculating CCR and MAE
			vccr(cnt) = ccrcalc(cm, size(testInputs,2));
			vmae(cnt) = maecalc(cm, size(testInputs,2));
			
			%% print the results in files
			
			% dataset CSV file
			fprintf(f1,'%d,%f,%f,%d,%f\n',cnt,vccr(cnt),vmae(cnt),vNHO(i-2),etime(clock, t));
			
			% results file
			fprintf(f,'Ejecucion %d\n',cnt);
			fprintf(f,'NHO: %d\n',vNHO(i-2));
			fprintf(f,'cm\n');
			for k=1:size(cm,1)
				for j=1:size(cm,2)
					fprintf(f,'%d ',cm(k,j));
				end
				fprintf(f,'\n');
			end
			fprintf(f,'ccr: %f\n',vccr(cnt));
			fprintf(f,'mae: %f\n',vmae(cnt));
			fprintf(f,'\n');
			
			cnt = cnt+1;
		end
	end
	% general CSV file
	fprintf(f2,'%s,%f,%f,%f,%f,%f,%f\n',dataset_name,mean(vccr),std(vccr),mean(vmae),std(vmae),mean(vNHO),std(vNHO));
end

fclose('all');

exit;
