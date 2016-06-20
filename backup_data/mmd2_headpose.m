%% Script to Perform Calculation of Beta in MMD formula

% Get a random list of numbers (participants)
all_subjects = {'p00/', 'p01/', 'p02/', 'p03/', 'p04/', 'p05/', 'p06/', 'p07/', 'p08/', 'p09/', 'p10/', 'p11/', 'p12/', 'p13/', 'p14/'};
rnd_list = randperm(15);
S_list = rnd_list(1:4);
T_list = rnd_list(5);

fprintf('Source: %d\n', S_list-1);
fprintf('Target: %d\n', T_list-1);

% Get Source and Target Image Data
S_all = getData(all_subjects(S_list), 20);
S_head = S_all.headpose;
fprintf('Got Source Data\n');

T_all = getData(all_subjects(T_list), 1);
fprintf('Got Target Data\n');

% Subsample the target data set
Percent_Sample = 1; % 10 percent
t = size(T_all.data);
T_all_length = t(4); 
T_das_length = int32(T_all_length*Percent_Sample/100);
T_das.data = T_all.data(:,:,:,1:T_das_length);
T_das.label = T_all.label(:,1:T_das_length);
T_das.head = T_all.headpose(:,1:T_das_length);

T_rest.data = T_all.data(:,:,:,T_das_length+1:end);
T_rest.label = T_all.label(:,T_das_length+1:end);
T_rest.head = T_all.headpose(:,T_das_length+1:end);

% Convert list of indexes to list of subjects
S_list = S_list - 1;
T_list = T_list - 1;

%% Find Sigma for Kernel Function (Gaussian)
fprintf('--- Sigma ---\n');
divider = 1;
size_s = size(S_head);
S_length = size_s(2);

runsum = 0;
for i=1:divider:S_length
    for j=1:divider:S_length
        runsum = runsum + distance(S_head(:,i), S_head(:,j));
    end
    %fprintf('Sigma Progress: %d of %d\n', i, S_length);
end
fprintf('--- Sigma Done ---\n');

% This is our sigma
avg_dist = runsum/((S_length^2)/(divider^2))

%% Find Ks,s - the array of kernel distances for all Source points to all other Source points
fprintf('--- Kss ---\n');
Kss = single(zeros(S_length)); 
parfor i=1:S_length
    for j=1:S_length
        Kss(i,j) = single(gaussianKernel(S_head(:,i), S_head(:,j), avg_dist));
    end
    %fprintf('Kss Progress: %d of %d\n', i, S_length);
end
fprintf('--- Kss Done ---\n');

%% Find Ks,l - the array of sum of distances to each element in target 
%size_t = size(T_imgs);
%T_length = size_t(4);

fprintf('--- Ksl ---\n')
ksl = zeros(1, S_length);
parfor s=1:S_length
    sum = 0;
    for t=1:T_das_length
        sum = sum + gaussianKernel(S_head(:,s), T_das.head(:,t), avg_dist);
    end
    ksl(s) = sum;
    %fprintf('Ksl Progress: %d of %d\n', s, S_length);
end
fprintf('--- Ksl Done ---\n')

lb = zeros(S_length, 1);
ub = ones(S_length, 1);
K = Kss*2/(S_length^2);
f = ksl*(-2/(S_length*double(T_das_length)));
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

% Write out weighted source data
trainname_weighted = strcat('Source_', int2str(S_list(1)), '_', int2str(S_list(2)), '_', int2str(S_list(3)), '_', int2str(S_list(4)), '_T', int2str(T_list(1)), '_weighted_hp.h5');
hdf5write(trainname_weighted,'/data', OutData.data, '/label',[OutData.label; OutData.headpose]); 

% Write out unweighted source data
trainname_unweighted = strcat('Source_', int2str(S_list(1)), '_', int2str(S_list(2)), '_', int2str(S_list(3)), '_', int2str(S_list(4)), '_T', int2str(T_list(1)), '_unweighted_hp.h5');
hdf5write(trainname_unweighted,'/data', S_all.data, '/label',[S_all.label; S_all.headpose]); 

% Write out subsample of target data
target_name_sub = strcat('Target_subject', int2str(T_list(1)), '_', int2str(Percent_Sample), 'percent_', 'hp.h5');
hdf5write(target_name_sub,'/data', T_das.data, '/label',[T_das.label; T_das.head]); 

% Write out the rest of the target data
target_name_rest = strcat('Target_subject', int2str(T_list(1)), '_', int2str(100-Percent_Sample), 'percent_', 'hp.h5');
hdf5write(target_name_rest,'/data', T_rest.data, '/label',[T_rest.label; T_rest.head]); 

% Write filenames to text file
fileID = fopen('filenames.txt','w');
pth = '/home/pauli/Gaze/bijcaffe/data/MPIIGaze/H5/';
fprintf(fileID, '%s\n', strcat(pth, trainname_weighted));
fprintf(fileID, '%s\n', strcat(pth, trainname_unweighted));
fprintf(fileID, '%s\n', strcat(pth, target_name_sub));
fprintf(fileID, '%s\n', strcat(pth, target_name_rest));
fclose(fileID); 