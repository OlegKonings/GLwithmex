m=int32(1920);%for testing only, just make sure for tests that dimensions of A match partition vector
K=int32(13);
density=single(100);

[ A,b,partition,lambda ] = GenerateRandomGroupLassoDataSet( m,K,density );

%NOTE: maximum number of lambdas is 31, and best to keep the number of
%lambdas less than smallest dimension of A

MAX_ITER = int32(100);
ABSTOL   = single(1e-4);
RELTOL   = single(1e-2);

alpha=single(1);
rho=single(1);

[m,n]=size(A);
m=int32(m);
n=int32(n);


do_obj=int32(0);
do_lam=int32(0);


lambda_array=single([.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0,1.1,1.2,1.3,1.4,1.7,1.9,2.1]');
num_lambdas=int32(length(lambda_array));
num_iters_gpu=int32(zeros(num_lambdas,1));
disp('num_lambdas= ');
disp(num_lambdas);
AA=A';% NOTE: since mex/CUDA call uses row-major, have to tranpose A before calling(will adjust [m,n] internally)
disp('partition sum= ');
dd=sum(partition);
disp(dd);
disp('m=');
disp(m);
disp('n=');
disp(n);
u=single(zeros(n,num_lambdas));
z=single(zeros(n,num_lambdas));

t=tic;
[nxtu,nxtz,num_iters_gpu]=GroupMextest(AA,b,partition,u,z,rho,alpha,MAX_ITER,ABSTOL,RELTOL,lambda_array,num_iters_gpu);% for this version matrix A must be passed in transpose (CUDA solver uses row major)

gtime=(toc(t));
%NOTE: The first call to mex file will be slow! MATLAB has to 'set-up'
%context and all subsquent calls will be at least 10x faster than intial
%call
disp('Full gpu time=');
disp(gtime);

t=tic;

x=single(zeros(n,num_lambdas));
u=single(zeros(n,num_lambdas));
z=single(zeros(n,num_lambdas));
% assuming that u and v are unchanged so will use them for both
% implementations
 
if (sum(partition) ~= n)
    error('invalid partition');
end
 
Atb = A'*b;
cum_part= int32(cumsum(double(partition)));

QUIET    = 1;
[L,U]=factor(A,rho);

has_converged = false(num_lambdas,1);
num_iterations_matlab=int32(zeros(num_lambdas,1));
for k = 1:MAX_ITER
    % x-update
    for kk=1:num_lambdas
        if has_converged(kk,1)
            continue;
        end
        q = Atb + rho*(z(:,kk) - u(:,kk));    % temporary value
        if( m >= n )    % if skinny
            x(:,kk) = U \ (L \ q);
        else            % if fat
            x(:,kk) = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
        end
        
        % z-update
        zold = z(:,kk);
        start_ind = 1;
        x_hat = alpha*x(:,kk) + (1-alpha)*zold;
        for i = 1:length(partition),
            sel = start_ind:cum_part(i);
            z(sel,kk) = shrinkage(x_hat(sel) + u(sel,kk), lambda_array(kk)/rho);
            start_ind = cum_part(i) + 1;
        end
        u(:,kk) = u(:,kk) + (x_hat - z(:,kk));
        
        % diagnostics, reporting, termination checks
        history.objval(k,kk)  = objective(A, b,lambda_array(kk), cum_part, x(:,kk),z(:,kk));
        
        history.r_norm(k,kk)  = norm(x(:,kk) - z(:,kk));
        history.s_norm(k,kk)  = norm(-rho*(z(:,kk) - zold));
        
        history.eps_pri(k,kk) = sqrt(single(n))*ABSTOL + RELTOL*max(norm(x(:,kk)), norm(z(:,kk) ));
        history.eps_dual(k,kk)= sqrt(single(n))*ABSTOL + RELTOL*norm(rho*u(:,kk));
        
%          fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
%             history.r_norm(k,kk), history.eps_pri(k,kk), ...
%             history.s_norm(k,kk), history.eps_dual(k,kk), history.objval(k,kk));
        
        if (history.r_norm(k,kk) <history.eps_pri(k,kk) && ...
                history.s_norm(k,kk) <history.eps_dual(k,kk))
            has_converged(kk,1)=true;
            num_iterations_matlab(kk,1)=k;
        end
        
    end

    if all(has_converged)
            break;
    end
   
end

ctime=(toc(t));
disp('Matlab multi lambda time= ');
disp(ctime);

disp('____________Error comparison of MATLAB vs. CUDAmex_______________');
for i=1:num_lambdas
    fprintf('lambda value=%f, Num iters gpu= %d, Num iters matlab= %d \n',lambda_array(i),num_iters_gpu(i,1),num_iterations_matlab(i,1));
    fprintf('norm u dif= %f ,',norm(u(:,i))-norm(nxtu(:,i)));
    fprintf('norm z dif= %f\n\n',norm(z(:,i))-norm(nxtz(:,i)));
end


    
