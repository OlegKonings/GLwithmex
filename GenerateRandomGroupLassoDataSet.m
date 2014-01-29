function [ A,b,partition,lambda ] = GenerateRandomGroupLassoDataSet( m,K,density )% density default value should be 100

flag=true;
while flag
    partition = int32(randi(128, [K 1]));% per Boyd example
    n = int32(sum(partition)); % number of features
    if (mod(n,16))==0
        flag=false;
    end
end

p = single(density/single(n));          % sparsity density

% generate block sparse solution vector
x = single(zeros(n,1));
start_ind = int32(1);
cum_part = int32(cumsum(single(partition)));

for i = 1:K,
    x(start_ind:cum_part(i)) = 0;
    if( rand() < p)
        % fill nonzeros
        x(start_ind:cum_part(i)) = randn(partition(i),1);
    end
    start_ind = cum_part(i)+1;
end

% generate random data matrix
A = randn(m,n);

% normalize columns of A
A = (A*spdiags(1./sqrt(sum(A.^2))',0,double(n),double(n)));
A=full(single(A));
% generate measurement b with noise
b = single(A*x + sqrt(0.001)*randn(m,1));

% lambda max
start_ind = 1;
lambdas=single(zeros(1,K+1));
for i = 1:K,
    sel = start_ind:cum_part(i);
    lambdas(i) = norm(A(:,sel)'*b);
    start_ind = cum_part(i) + 1;
end
lambda_max = max(lambdas);

% regularization parameter
lambda = 0.1*lambda_max;


end

