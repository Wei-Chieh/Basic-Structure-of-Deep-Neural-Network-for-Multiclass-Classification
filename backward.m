function [model, sW, dW] = backward(model, sW, dW, AA, grad, eta, alpha, momentum)
    l = length(model);
    [d, n] = size(grad);
    eta1 = eta / (d * n);

    for i = l:-1:1
        if i == l, QQ = grad;
        else QQ = (model{i + 1}.W.' * DD{i + 1});
        end
        switch model{i}.f
            case 'sigmoid'
                GG = AA{i + 1} .* (1 - AA{i + 1});
            case 'ReLU'
                GG = (AA{i + 1} < 0) * 0.2 + (AA{i + 1} > 0) * 1;
            case 'linear'
                GG = 1;
        end
        DD{i} = QQ .* GG;
        if i ~= l, DD{i}(end, :) = []; end
    end
    
    [model, sW, dW] = cellfun(@(dd, aa, m, sw, dw) ...
        prop(dd, aa, m, sw, dw, eta1, alpha, momentum), ...
        DD, AA(1:end-1), model, sW, dW, 'UniformOutput', false);
end

function [m, sw, dw] = prop(DD, AA, M, sW, dW, eta, alpha, momentum)
    gW = DD * AA.';
    if sum(sum(sW)) == 0, sW = gW; end
    sw = sqrt((sW .^ 2) * alpha + (gW .^ 2) * (1 - alpha));
    delta = ((gW + dW) ./ sw) * eta;
    delta(isnan(delta)) = 0;
    m.W = M.W - delta;
    m.f = M.f;
    dw = gW * momentum;
end