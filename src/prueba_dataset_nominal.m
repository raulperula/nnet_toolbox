%%% uso del conjunto de datos balance en redes neuronales

% carga del conjunto de datos para train y test
loaddata;

%% uso del conjunto de datos de entrenamiento
% creacion de la red neuronal y ajustada para la parte ordinal
net = newff(trainInputs,trainTargets,20,{'logsig','logsig'},'trainrp');

% entrenamiento de la red
net = train(net,trainInputs,trainTargets);

%% uso del conjunto de datos de testeo

% simulacion de la red para obtener las salidas
testOutputs = sim(net,testInputs);

% tratamiento de los valores de salida
aux = zeros(size(testOutputs));
[c,i] = max(testOutputs);
j = 1;
for k=i
	aux(k,j) = 1;
	j = j+1;
end
testOutputs = aux;

%% medidas de error
% calculo de la matriz de confusion
[c,cm,ind,per] = confusion(testTargets,testOutputs);
plotconfusion(testTargets,testOutputs)

% calculo de CCR y MAE
ccrcalc(cm, size(testInputs,2))
maecalc(cm, size(testInputs,2))
