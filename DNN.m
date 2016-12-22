function fM = DNN(XTr, YTr, XV, YV, parameters)
    eta = parameters.eta;
    batchsize = parameters.batchsize;
    decayRate = parameters.decayRate;
    momentum = parameters.momentum;
    alpha = parameters.alpha;
    maxEpoch = parameters.maxEpoch;
    layers = parameters.layers;
    funcs = parameters.funcs;

    layers = [size(XTr, 1), layers, size(YTr, 1)];
    [model, sW, dW] = init(layers, funcs);
    AA = forward(XV, model);
    fprintf('Initialized, ');
    Acc = evaluate(AA{end}, YV);
    
    fM = model;
    fA = Acc;
    
    for iter = 1:maxEpoch
        n = size(XTr, 2);
        lalala = randperm(n);
        for i = 1:floor(n / batchsize)
            lala = lalala(((i - 1) * batchsize + 1):(i * batchsize));
            Data = XTr(:, lala);
            Ref = YTr(:, lala);
            
            if gpuDeviceCount>0
                Data = gpuArray(Data);
                Ref = gpuArray(Ref);
            end
            
            AA = forward(Data, model);
            [~, grad] = errfunc(AA{end}, Ref);
            [model, sW, dW] = backward(model, sW, dW, AA, grad, eta, alpha, momentum);
        end
        eta = eta * decayRate;
        
        AA = forward(XV, model);
        fprintf('Iter = %d, ', iter);
        Acc = evaluate(AA{end}, YV);
        if Acc > fA
            fM = model;
            fA = Acc;
        end
    end
end