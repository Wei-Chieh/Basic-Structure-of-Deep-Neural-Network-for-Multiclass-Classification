function [Acc, Err] = evaluate(P, Y)
    [~, pi] = max(P);
    [~, yi] = max(Y);
    Acc = mean(pi == yi);
    Err = errfunc(P, Y);
    fprintf('Accuracy = %f, Error = %f\n', Acc, Err);
end