%% Script to Perform Calculation of Beta in MMD formula

% Get Source and Target Image Data
S_all = getData({'p00/', 'p001'}, 11);
S_imgs = S_all.data;
fprintf('Got Source Data\n');

T_all = getData({'p14/'}, 11);
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






%{
flat_src_sub = flat_src(:, 1:11:end);
s = size(flat_src_sub);
l = s(2);

runsum = 0;
for i=1:l
    for j=1:l
        runsum = runsum + distance(flat_src(:,i), flat_src(:,j));
    end
    fprintf('Progress: %d of %d\n', i, l);
end

epsilon = runsum/(l*l)

%%

%SourceTarget = cat(4, Source, Target);


%for j=0:600:length(Source)
%    for i=0:600:length(Source)
%        Kss(i/600+1, j/600+1) = gaussianKernel(Source(:,:,:,i/600+1), Source(:,:,:,j/600+1), 1);
%    end
%    fprintf('Progress: %d of %d\n', j, length(Source));
%end

%}