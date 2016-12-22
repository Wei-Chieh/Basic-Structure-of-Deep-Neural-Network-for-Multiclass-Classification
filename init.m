function [model, sW, dW] = init(layers, funcs)
    for i = 1:length(layers) - 1
        model{i}.W = rand(layers(i + 1), layers(i) + 1) * 0.2 - 0.1;
        model{i}.f = funcs{i};
        sW{i} = zeros(layers(i + 1), layers(i) + 1);
        dW{i} = sW{i};
    end
    for i = 1:length(layers) - 1
        model{i}.W = single(model{i}.W);
        sW{i} = single(sW{i});
        dW{i} = single(dW{i});
    end
    if gpuDeviceCount>0
        for i = 1:length(layers) - 1
            model{i}.W = gpuArray(model{i}.W);
            sW{i} = gpuArray(sW{i});
            dW{i} = gpuArray(dW{i});
        end
    end
end