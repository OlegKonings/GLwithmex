function [L,U ] = factor(A, rho)
[m, n] = size(A);
    if ( m >= n )   
       L = chol( A'*A + rho*eye(n), 'lower' );
    else        
       L = chol( eye(m) + 1/rho*(A*A'), 'lower' );
    end
    U = L';
end

