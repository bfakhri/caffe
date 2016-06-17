%% Script to Perform Calculation of Beta in MMD formula

% Get a random list of numbers (participants)
all_subjects = {'p00/', 'p01/', 'p02/', 'p03/', 'p04/', 'p05/', 'p06/', 'p07/', 'p08/', 'p09/', 'p10/', 'p11/', 'p12/', 'p13/', 'p14/'};
rnd_list = randperm(15);
S_list = rnd_list(1:4);
T_list = rnd_list(5);

fprintf('Source: %d\n', S_list-1);
fprintf('Target: %d\n', T_list-1);

% Get Source and Target Image Data
S_all = getData(all_subjects(S_list), 73);
S_imgs = S_all.data;
fprintf('Got Source Data\n');

T_all = getData(all_subjects(T_list), 1);
fprintf('Got Target Data\n');

% Subsample the target data set
Percent_Sample = 10; % 10 percent
t = size(T_all.data);
T_all_length = t(4); 
T_das_length = int32(T_all_length*Percent_Sample/100);
T_das.data = T_all.data(:,:,:,1:T_das_length);

%% Find Sigma for Kernel Function (Gaussian)
divider = 213;
size_s = size(S_imgs);
S_length = size_s(4);
S_imgs_flat = reshape(S_imgs, [60*36, S_length]);
runsum = 0;
for i=1:divider:S_length
    for j=1:divider:S_length
        runsum = runsum + distance(S_imgs_flat(:,i), S_imgs_flat(:,j));
    end
    fprintf('Sigma Progress: %d of %d\n', i, S_length);
end

% This is our sigma
avg_dist = runsum/((S_length^2)/(divider^2))

%% Find Ks,s - the array of kernel distances for all Source points to all other Source points
Kss = zeros(S_length); 
for i=1:S_length
    for j=1:S_length
        Kss(i,j) = gaussianKernel(S_imgs_flat(:,i), S_imgs_flat(:,j), avg_dist);
    end
    fprintf('Kss Progress: %d of %d\n', i, S_length);
end

%% Find Ks,l - the array of sum of distances to each element in target 
size_t = size(T_imgs);
T_length = size_t(4);

ksl = zeros(1, S_length);
for s=1:S_length
    sum = 0;
    for t=1:T_das_length
        sum = sum + gaussianKernel(S_imgs(:,s), T_das.data(:,t), avg_dist);
    end
    ksl(s) = sum;
    fprintf('Ksl Progress: %d of %d\n', s, S_length);
end

lb = zeros(S_length, 1);
ub = ones(S_length, 1);
K = Kss*2/(S_length^2);
f = ksl*(-2/(S_length*T_length));
beta = quadprog(double(K), double(f), [], [], [], [], lb, ub);
threshold = mean(beta); 

%% Save data out to a file
OutData=[];

count = 1;
for i=1:S_length
    if(beta(i) > threshold)
        OutData.data(:,:,:,count) = S_all.data(:,:,:,i);
        OutData.label(:,count) = S_all.label(:,i);
        OutData.headpose(:,count) = S_all.headpose(:,i);
        count = count + 1;
    end
end

% Convert list of indexes to list of subjects
S_list = S_list - 1;
T_list = T_list - 1;

trainname = strcat('Source_', int2str(S_list(1)), '_', int2str(S_list(2)), '_', int2str(S_list(3)), '_', int2str(S_list(4)), '.h5');
% Write out
hdf5write(trainname,'/data', OutData.data, '/label',[OutData.label; OutData.headpose]); 

testname = strcat('Target_subject', int2str(T_list(1)), '_', int2str(Percent_Sample), 'percent_', 'samples.h5');
hdf5write(testname,'/data', T_all.data, '/label',[T_all.label; T_all.headpose]); 