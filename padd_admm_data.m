function [ A,b,u,z ] = padd_admm_data( oldA,oldb,oldu,oldz )
%padd_admm_data pads input data for admm with zeros 
%vector oldb is of length oldM, vectors oldu,oldz are of length oldN
[oldM,oldN]=size(oldA);
if( (mod(oldM,16)~=0) || (mod(oldN,16)~=0))
    
    m=16*(int32(oldM+16-1)/int32(16));
    n=16*(int32(oldN+16-1)/int32(16));
    %disp(m);
    %disp(n);
    A=zeros(m,n);
    A(1:oldM,1:oldN)=oldA;
    
    b=zeros(m,1);
    u=zeros(n,1);
    z=zeros(n,1);
    
    b(1:oldM)=oldb;
    u(1:oldN)=oldu;
    z(1:oldN)=oldz;
else
    A=oldA;
    b=oldb;
    u=oldu;
    z=oldz;
    
end

end

