%% Script to Perform Calculation of Beta in MMD formula

% Get Source and Target Image Data
S_all = getData({'p00/'});
S_imgs = S_all.data;
fprintf('Got Source Data\n');

T_all = getData({'p14/'});
T_imgs = T_all.data;
fprintf('Got Target Data\n');

%% Find Sigma for Kernel Function (Gaussian)

divider = 11;
size_s = size(S_imgs);
length = size_s(4);
S_imgs_flat = reshape(S_imgs, [60*36, length]);
runsum = 0;
for i=1:divider:length
    for j=1:divider:length
        runsum = runsum + distance(S_imgs_flat(:,i), S_imgs_flat(:,j));
    end
    fprintf('Progress: %d of %d\n', i*divider, length/divider);
end
avg_dist = runsum/((length^2)/(divider^2))









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