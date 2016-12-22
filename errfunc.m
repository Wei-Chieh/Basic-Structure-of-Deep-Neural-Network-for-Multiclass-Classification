function [err, grad] = errfunc(P, Y)
    PP = exp(P);
    PPP = bsxfun(@rdivide, PP, sum(PP));
    err = mean(-sum(Y .* log(PPP)));
    grad = PPP - Y;
end