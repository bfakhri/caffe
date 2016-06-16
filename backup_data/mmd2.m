%% Script to Perform Calculation of Beta in MMD formula

% Get Source and Target Image Data
S_all = getData({'p00/', 'p01/', 'p02/', 'p03/', 'p04/', 'p05/', 'p06/', 'p07/', 'p08/'}, 71);
S_imgs = S_all.data;
fprintf('Got Source Data\n');

T_all = getData({'p14/'}, 71);
T_imgs = T_all.data;
fprintf('Got Target Data\n');

%% Find Sigma for Kernel Function (Gaussian)

divider = 1;
size_s = size(S_imgs);
S_length = size_s(4);
S_imgs_flat = reshape(S_imgs, [60*36, S_length]);
runsum = 0;
for i=1:divider:S_length
    for j=1:divider:S_length
        runsum = runsum + distance(S_imgs_flat(:,i), S_imgs_flat(:,j));
    end
    fprintf('Progress: %d of %d\n', i*divider, S_length/divider);
end
% This is our sigma
avg_dist = runsum/((S_length^2)/(divider^2))

%% Find Ks,s - the array of kernel distances for all Source points to all other Source points
for i=1:S_length
    for j=1:S_length
        Kss(i,j) = gaussianKernel(S_imgs_flat(:,i), S_imgs_flat(:,j), avg_dist);
    end
    fprintf('Progress: %d of %d\n', i, S_length);
end

%% Find Ks,l - the array of sum of distances to each element in target 
size_t = size(T_imgs);
T_length = size_t(4);

for s=1:S_length
    sum = 0;
    for t=1:T_length
        sum = sum + gaussianKernel(S_imgs(:,s), T_imgs(:,t), avg_dist);
    end
    ksl(s) = sum;
    fprintf('Progress: %d of %d\n', s, S_length);
end


A = zeros(S_length);
b = zeros(S_length, 1);
Aeq= ones(S_length,1);
beq = 500;
lb = zeros(S_length, 1);
ub = ones(S_length, 1);
%beta = quadprog(double(Kss), double(ksl))
%beta = quadprog(double(Kss), double(ksl), A, b, transpose(Aeq), beq, lb, ub);
K = Kss*2/(S_length^2);
f = ksl*(-2/(S_length*T_length));
beta = quadprog(double(K), double(f), [], [], [], [], lb, ub);
beta_norm = (beta-min(beta))/(max(beta)-min(beta));

%% Save data out to a file
%{
OutData=[];
%OutData.data = zeros(60,36,1, total_num*2);
%OutData.label = zeros(2, total_num*2);
%OutData.headpose = zeros(2, total_num*2);
%OutData.confidence = zeros(1, total_num*2);

count = 1;
for i=1:S_length
    if(beta(i) > 0.5)
        OutData.data(:,:,:,count) = S_all.data(:,:,:,i);
        OutData.label(:,count) = S_all.label(i);
        OutData.label(count*2) = S_all.label(i+1);
        OutData.headpose(count*2-1) = S_all.headpose(i);
        OutData.headpose(count*2) = S_all.headpose(i+1);
    end
end

savename = 'mmd_selected_samples.h5';
% Write out
hdf5write(savename,'/data', OutData.data, '/label',[OutData.label; OutData.headpose]); 
%}