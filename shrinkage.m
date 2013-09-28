function z = shrinkage(x, kappa)
z = subplus(1 - kappa/norm(x))*x;
end

