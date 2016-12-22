clear
parameters.dataset = 'MNIST';
parameters.eta = 0.05;
parameters.batchsize = 100;
parameters.decayRate = 0.95;
parameters.momentum = 0.8;
parameters.alpha = 0.9;
parameters.maxEpoch = 30;
parameters.layers = [500, 500];
parameters.funcs = {'ReLU', 'ReLU', 'linear'};

load(parameters.dataset);
model = DNN(XTr, YTr, XV, YV, parameters);

AA = forward(XTe, model);
evaluate(AA{end}, YTe);