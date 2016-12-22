function  AA = forward(XX, model)
    l = size(model, 2);
    n = size(XX, 2);
    AA{1} = [XX; ones(1, n)];
    for i = 1:l
        QQ = model{i}.W * AA{i};
        switch model{i}.f
            case 'sigmoid'
                GG = sigmf(QQ, [1 0]);
            case 'ReLU'
                GG = ((QQ < 0) * 0.2 + (QQ > 0) * 1) .* QQ;
            case 'linear'
                GG = QQ;
        end
        if i == l, AA{i + 1} = GG;
        else AA{i + 1} = [GG; ones(1, n)];
        end
    end
end