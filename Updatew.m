function [w wobj]=Updatew(E,H,vN,eta,lambda)
f=[];
for wIndex=1:vN
    vH=2*lambda*H;
    Etemp=E{wIndex};
%     tempSum=sum(sqrt(sum(Etemp.^2,1)));
tempSum=norm(Etemp,'fro')*norm(Etemp,'fro');
    f=[f;eta*tempSum];
end
beq=1;
aeq=ones(1,vN);
lb=zeros(vN,1);
[w,wobj]= quadprog(H,f,[],[],aeq,beq,lb);