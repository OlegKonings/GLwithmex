function [ z ] = MySubPlus( x )
z=x;
[len]=size(x);
for k=1:len
    z(k)=max(x(k),0); 
end

end

