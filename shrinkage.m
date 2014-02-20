function z = shrinkage(x, kappa)
z = MySubPlus(1 - kappa/norm(x))*x;
end

