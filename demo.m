%% main function
%The code is for paper "RGB-T Saliency Detection via Low-rank Tensor Learning and Unified Collaborative Ranking" 
%Writen by Liming Huang.

%% setting
clear;
imgRoot='./Img/RGBT/';
imgRoot1='./Img/RGB1000/'; % RGB images input
imgRoot2='./Img/T1000/'; % Thermal images input
saldir='./sal_map/1000/'; % the output path of the saliency map
supdir='./superpixels/'; % the superpixel label file path
FCNfeatureRoot1 = './FCN-feature/RGB1000/'; % use pretrained FCN-32S network
FCNfeatureRoot2 = './FCN-feature/T1000/';
mkdir(supdir);
mkdir(saldir);
imnames1=dir([imgRoot1 '*' 'jpg']);
imnames2=dir([imgRoot2 '*' 'jpg']);
theta1=20;
theta2=40;
theta3=20;
theta4=40;
spnumber=300;
pi=1.8;
%%
 for ii=1:length(imnames1)  
     disp(ii);
    im1=[imgRoot1 imnames1(ii).name];
    im2=[imgRoot2 imnames2(ii).name];     
    img1=imread(im1);
    img2=imread(im2);   
    Simg=0.5*img1 + 0.5*img2; % generating RGBT image 
    Simgn=[imgRoot imnames1(ii).name];
    Simgname=[Simgn(1:end-4)  '.bmp'];
    imwrite(Simg,Simgname,'bmp');
    [m,n,k]=size(Simg);
   
 %%    SLICSuperpixelSegmentation 
     Simgname=[Simgn(1:end-4)  '.bmp'];
     comm=['SLICSuperpixelSegmentation' ' ' Simgname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];
     system(comm); % SLIC
     spname=[supdir imnames1(ii).name(1:end-4)  '.dat'];
     superpixels=ReadDAT( [m,n],spname);
     spnum=max(superpixels(:));
 %%    Edges               
     adjloop=AdjcProcloop(superpixels,spnum);
     edges=[];
     for i=1:spnum
         indext=[];
         ind=find(adjloop(i,:)==1);
         for j=1:length(ind)
             indj=find(adjloop(ind(j),:)==1);
             indext=[indext,indj];
         end
         indext=[indext,ind];
         indext=indext((indext>i));
         indext=unique(indext);
         if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=i*ed(:,2);
            ed(:,1)=indext;
            edges=[edges;ed];
        end
     end
     inds = cell(spnum,1);
     for ttt=1:spnum
         inds{ttt} = find(superpixels==ttt);
     end
   
    %%    compute RGB modality FCN affinity matrix 
    [G_meanVgg1,G_meanVgg2] = ExtractFCNfeature(FCNfeatureRoot1,imnames1(ii).name(1:end-4),inds,m,n);        
    G_weights1 = makeweights(edges,G_meanVgg1,theta1);    
    G_weights2 = makeweights(edges,G_meanVgg2,theta1);   
    G_W1 = adjacency(edges,G_weights1,spnum); 
    G_W2 = adjacency(edges,G_weights2,spnum);

     %%    compute thermal modality FCN affinity matrix
    [T_meanVgg1,T_meanVgg2] =ExtractFCNfeature(FCNfeatureRoot2,imnames2(ii).name(1:end-4),inds,m,n);
    T_weights1 = makeweights(edges,T_meanVgg1,theta2);    
    T_weights2 = makeweights(edges,T_meanVgg2,theta2);   
    T_W1 = adjacency(edges,T_weights1,spnum); 
    T_W2 = adjacency(edges,T_weights2,spnum);
   
    %%    CIE-LAB color feature
   input_vals1 = reshape(img1, m*n, k); 
   input_vals2 = reshape(img2, m*n, k);
   rgb_vals1 = zeros(spnum,1,3);
   rgb_vals2 = zeros(spnum,1,3); 
   for i = 1:spnum
        rgb_vals1(i,1,:) = mean(input_vals1(inds{i},:),1);  
        rgb_vals2(i,1,:) = mean(input_vals2(inds{i},:),1);
   end
   lab_vals1 = colorspace('Lab<-', rgb_vals1);
   lab_vals2 = colorspace('Lab<-', rgb_vals2);
   seg_vals1=reshape(lab_vals1,spnum,3);% lab 颜色特征   
   seg_vals2=reshape(lab_vals2,spnum,3);% lab 颜色特征
   Weights1 = makeweights(edges,seg_vals1,theta3); 
   Weights2 = makeweights(edges,seg_vals2,theta4); 
   G_W3 = adjacency(edges,Weights1,spnum);  
   T_W3 = adjacency(edges,Weights2,spnum);     
   
   %% low-rank tensor learning   
   A1=zeros(spnum,spnum,3);
   A2=zeros(spnum,spnum,3);
   A1(:,:,1)=G_W1;
   A1(:,:,2)=G_W2;
   A1(:,:,3)=G_W3;
   A2(:,:,1)=T_W1;
   A2(:,:,2)=T_W2;
   A2(:,:,3)=T_W3;
   tau = 0.1;
   
   [Z1,Z2] = LR_Tensor(A1,A2,tau);

   G_W1=Z1(:,:,1);
   dd = sum(G_W1); G_D1 = sparse(1:spnum,1:spnum,dd); clear dd;
   G_L1 =G_D1-G_W1; 
   
   G_W2=Z1(:,:,2);
   dd = sum(G_W2); G_D2 = sparse(1:spnum,1:spnum,dd); clear dd;
   G_L2 =G_D2-G_W2;
   
   G_W3=Z1(:,:,3);
   dd = sum(G_W3); G_D3 = sparse(1:spnum,1:spnum,dd); clear dd;
   G_L3 =G_D3-G_W3;
   
   T_W1=Z2(:,:,1);
   dd = sum(T_W1); T_D1 = sparse(1:spnum,1:spnum,dd); clear dd;
   T_L1 =T_D1-T_W1;
   
   T_W2=Z2(:,:,2);
   dd = sum(T_W2); T_D2 = sparse(1:spnum,1:spnum,dd); clear dd;
   T_L2 =T_D2-T_W2;
   
   T_W3=Z2(:,:,3);
   dd = sum(T_W3); T_D3 = sparse(1:spnum,1:spnum,dd); clear dd;
   T_L3 =T_D3-T_W3;
   
 %%    Calculate H 
    WG=cell(1,3);
    WT=cell(1,3);
    DG=cell(1,3);
    DT=cell(1,3);
    TransitionG=cell(1,3);
    TransitionT=cell(1,3);
    WG{1}=G_W1;WG{2}=G_W2;WG{3}=G_W3;   WT{1}=T_W1;WT{2}=T_W2;WT{3}=T_W3;
    DG{1}=G_D1;DG{2}=G_D2;DG{3}=G_D3;   DT{1}=T_D1;DT{2}=T_D2;DT{3}=T_D3;
    
     for vIndex=1:3
        WG1 = WG{vIndex};
        DG1 = DG{vIndex};
        TransitionG{vIndex}=inv(DG1)*WG1;
     end
    
    for vIndex=1:3
        WT1 = WT{vIndex};
        DT1 = DT{vIndex};
        TransitionT{vIndex}=inv(DT1)*WT1;
    end
   % Calculate H
    Transition=cell(1,6);
    Transition{1}=TransitionG{1};Transition{2}=TransitionG{2};Transition{3}=TransitionG{3};
    Transition{4}=TransitionT{1};Transition{5}=TransitionT{2};Transition{6}=TransitionT{3};
    H=zeros(6,6);
    for Hvi=1:6
        for Hvj=1:6
            H(Hvi,Hvj)=trace(Transition{Hvi}'*Transition{Hvj});
        end
    end

    A =cell(1,6);
    A{1}=G_W1;A{2}=G_W2;A{3}=G_W3;  
    A{4}=T_W1;A{5}=T_W2;A{6}=T_W3;
        %% ------------------------ranking stage1----------------------%%      
    %% top
     Yt=zeros(spnum,1);
     bst=unique(superpixels(1,1:n));         
     Yt(bst)=1;
     [St] =LTCR(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yt,A,H);   %ranking with top boundary queries
     St=(St-min(St(:)))/(max(St(:))-min(St(:)));
     St=1-St;  

    %% down
     Yd=zeros(spnum,1);
     bst=unique(superpixels(m,1:n));         
     Yd(bst)=1;
     [Sd] =LTCR(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yd,A,H);    %ranking with down boundary queries
     Sd=(Sd-min(Sd(:)))/(max(Sd(:))-min(Sd(:)));
     Sd=1-Sd;
    %% right
     Yr=zeros(spnum,1);
     bst=unique(superpixels(1:m,1));         
     Yr(bst)=1;
     [Sr] =LTCR(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yr,A,H);    %ranking with right boundary queries
     Sr=(Sr-min(Sr(:)))/(max(Sr(:))-min(Sr(:)));
     Sr=1-Sr; 
    %% left
     Yl=zeros(spnum,1);
     bst=unique(superpixels(1:m,n));         
     Yl(bst)=1;
     [Sl] =LTCR(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,Yl,A,H);    %ranking with left boundary queries
     Sl=(Sl-min(Sl(:)))/(max(Sl(:))-min(Sl(:)));
     Sl=1-Sl;
   %% combine 
    Sc=(St.*Sd.*Sl.*Sr);
    Sc=(Sc-min(Sc(:)))/(max(Sc(:))-min(Sc(:))); 
  %% show the result of  stage1          
%     mapstage1=zeros(m,n);
%     for i=1:spnum
%         mapstage1(inds{i})=Sc(i);
%     end
%     mapstage1=(mapstage1-min(mapstage1(:)))/(max(mapstage1(:))-min(mapstage1(:)));
%     mapstage1=uint8(mapstage1*255);  
%     outname=[saldir imnames1(ii).name(1:end-4) '_stage1.png'];
%     imwrite( mapstage1 , outname); 
   %% foreground seeds
     seeds=Sc;
     threhold=mean(Sc)*pi;
     seeds(seeds<threhold)=0;
     seeds(seeds>=threhold)=1;   %generate foreground queries
    %% ------------------------stage2----------------------%%   
    [S] =LTCR(G_L1,G_L2,G_L3,T_L1,T_L2,T_L3,spnum,seeds,A,H);  %ranking with foreground queries
    mapstage2=zeros(m,n);
    for i=1:spnum
        mapstage2(inds{i})=S(i);
    end
    mapstage2=(mapstage2-min(mapstage2(:)))/(max(mapstage2(:))-min(mapstage2(:)));
    mapstage2=uint8(mapstage2*255);  
    outname=[saldir imnames1(ii).name(1:end-4) '.png'];
    imwrite(mapstage2 , outname); 
 end

 