%% Script to Create Datasets for Domain Adaptation based on MMD Calculations

%% Randomize Participants to Get Data From 'DA' - DA Group 'T' - Test Subject
all_subjects = {'p00/', 'p01/', 'p02/', 'p03/', 'p04/', 'p05/', 'p06/', 'p07/', 'p08/', 'p09/', 'p10/', 'p11/', 'p12/', 'p13/', 'p14/'};
%rnd_list = randperm(15);
%DA_list = rnd_list(1:4) - 1;
%T_list = rnd_list(7) - 1;

% Specific test subjects
DA_list = [9 8 14 5];
T_list = [0];

fprintf('DA Group: %d\n', DA_list);
fprintf('Test Subject: %d\n', T_list);

%% Get Data from MPIIGaze Data Set

DA_All = getData(all_subjects(DA_list+1), 20);
T_All = getData(all_subjects(T_list+1), 20);

for Percent_Sample= [5 20 30]
    fprintf('\nStarting: %d\n', Percent_Sample);
    % Split Test Subject into several parts
    len_da = size(DA_All.data, 4);
    len_t = size(T_All.data, 4); 
    len_t_half = idivide(len_t, int32(2), 'floor');

    len_t_percent = floor(len_t*Percent_Sample/100); % Percentage Here!!!

    % Get training data from test subject
    T_train.data = T_All.data(:,:,:,1:len_t_half);
    T_train.label = T_All.label(:,1:len_t_half);
    T_train.headpose = T_All.headpose(:,1:len_t_half);

    % Split training data into the da sample and the rest
    T_train_da.data = T_train.data(:,:,:,1:len_t_percent);
    T_train_da.label = T_train.label(:,1:len_t_percent);
    T_train_da.headpose = T_train.headpose(:,1:len_t_percent);

    T_train_rest.data = T_train.data(:,:,:,len_t_percent+1:end);
    T_train_rest.label = T_train.label(:,len_t_percent+1:end);
    T_train_rest.headpose = T_train.headpose(:,len_t_percent+1:end);

    % The latter half of the test subject data becomes the test data
    T_test.data = T_All.data(:,:,:,len_t_half+1:end);
    T_test.label = T_All.label(:,len_t_half+1:end);
    T_test.headpose = T_All.headpose(:,len_t_half+1:end);

    % Print sizes of arrays
    s = size(DA_All.data, 4);
    fprintf('--- DA_All.data: %d ---\n', s);
    s = size(T_All.data, 4);
    fprintf('--- T_All.data: %d ---\n', s);
    s = size(T_train.data, 4);
    fprintf('--- T_train.data : %d ---\n', s);
    s = size(T_train_da.data, 4);
    fprintf('--- T_train_da.data : %d ---\n', s);
    s = size(T_train_rest.data, 4);
    fprintf('--- T_train_rest.data : %d ---\n', s);

    %% Find Sigma for Kernel Functions
    fprintf('--- Calculating Sigma ---\n');
    sigma_data = 0;
    sigma_headpose = 0;
    stride = int32(5);

    flat_data = reshape(DA_All.data, 60*36, []);
    for i=1:stride:len_da
        for j=1:stride:len_da
            sigma_data = sigma_data + distance(flat_data(:, i), flat_data(:,j));
            sigma_headpose = sigma_headpose + distance(DA_All.headpose(:,i), DA_All.headpose(:,j));
        end
    end
    sigma_data = sigma_data/((len_da^2)/(double(stride)^2));
    sigma_headpose = sigma_headpose/((len_da^2)/(double(stride)^2));
    %fprintf('--- Done ---\n');

    %% Find Ks,s - array of kernel distance from all DA source points to all other DA source points
    fprintf('--- Calculating Ks,s ---\n');
    Kss_data = single(zeros(len_da));
    Kss_headpose = single(zeros(len_da));

    parfor i=1:len_da
        for j=1:len_da
            Kss_data(i,j) = single(gaussianKernel(flat_data(:,i), flat_data(:,j), sigma_data));
            Kss_headpose(i,j) = single(gaussianKernel(DA_All.headpose(:,i), DA_All.headpose(:,j), sigma_headpose));
        end
    end
    %fprintf('--- Done ---\n');

    %% Find ks,l - vector of distance from all DA elements to all test subject elements
    fprintf('--- Calculating ks,l ---\n');
    ksl_data = zeros(1, len_da);
    ksl_headpose = zeros(1, len_da);

    flat_data_ts = reshape(T_train_da.data, 60*36, []);

    len_t = size(T_train_da.data, 4);
    parfor s=1:len_da
        data_sum = 0;
        headpose_sum = 0;
        for t=1:len_t
            data_sum = data_sum + gaussianKernel(flat_data(:,s), flat_data_ts(:,t), sigma_data);
            headpose_sum = headpose_sum + gaussianKernel(DA_All.headpose(:,s), T_train_da.headpose(:,t), sigma_headpose);
        end
        ksl_data(s) = data_sum;
        ksl_headpose(s) = headpose_sum;
    end
    %fprintf('--- Done ---\n');

    %% Send Data to QP Solver to get Best (with respect to test subject) Samples to train on
    fprintf('--- Solving QP Problem ---\n');

    lb = zeros(len_da, 1);
    ub = ones(len_da, 1);
    K_data = Kss_data*2/(len_da^2);
    K_headpose = Kss_headpose*2/(len_da^2);
    f_data = ksl_data*(-2/(len_da*len_t));
    f_headpose = ksl_headpose*(-2/(len_da*len_t));
    
    % Calculate the Betas
    Beta_data = quadprog(double(K_data), double(f_data), [], [], [], [], lb, ub);
    Beta_headpose = quadprog(double(K_headpose), double(f_headpose), [], [], [], [], lb, ub);
    % Normalize the Betas
    Beta_data_norm = (Beta_data - min(Beta_data))/(max(Beta_data)-min(Beta_data));
    Beta_headpose_norm = (Beta_headpose - min(Beta_headpose))/(max(Beta_headpose)-min(Beta_headpose));
    


    %% Write out data to several files
    fprintf('--- Writing Out to Files ---\n');

    % The names of the files are stored here
    fileID = fopen(strcat(int2str(Percent_Sample),'p_train_list.txt'),'w');
    pth = '/home/pauli/Gaze/bijcaffe/data/MPIIGaze/H5/';

    % DAG Weighted Samples
    filename = strcat(int2str(Percent_Sample), 'p_DAG_weighted_', int2str(DA_list(1)), '_', int2str(DA_list(2)), '_', int2str(DA_list(3)), '_', int2str(DA_list(4)), '_', '_TS_', int2str(T_list(1)), '.h5');
    fprintf(fileID, '%s\n', strcat('./', filename));
    hdf5write(filename,'/data', DA_All.data, '/label',[DA_All.label; DA_All.headpose], '/data_weights', Beta_data_norm, '/hp_weights', Beta_headpose_norm, '/added_weights', Beta_data_norm+Beta_headpose_norm, '/ones', ones(1, size(DA_All.data, 4)); 

    % Test Subject DA training subset
    filename = strcat(int2str(Percent_Sample), 'p_T_train_A', '_TS_', int2str(T_list(1)), '.h5');
    fprintf(fileID, '%s\n', strcat('./', filename));
    hdf5write(filename,'/data', T_train_da.data, '/label',[T_train_da.label; T_train_da.headpose], '/data_weights', ones(1, size(T_train_da.data, 4)));
    %hdf5write(filename,'/data', T_train_da.data, '/label',[T_train_da.label; T_train_da.headpose], '/data_weights', ones(1, size(T_train_da.data, 4)), '/hp_weights', ones(1, size(T_train_da.data, 4))); 
    fclose(fileID);
    
    % Test Subject rest leftover of training data after DA training subset
    %filename = strcat('T_train_B', int2str(100-Percent_Sample), 'percent_', 'TS_', int2str(T_list(1)), '.h5');
    %fprintf(fileID, '%s\n', strcat(pth, filename));
    %hdf5write(filename,'/data', T_train_rest.data, '/label',[T_train_rest.label; T_train_rest.headpose], '/data_weights', ones(1, size(T_train_rest.data, 4))); 
    %hdf5write(filename,'/data', T_train_rest.data, '/label',[T_train_rest.label; T_train_rest.headpose], '/data_weights', ones(1, size(T_train_rest.data, 4)), '/hp_weights', ones(1, size(T_train_rest.data, 4))); 

    fileID = fopen(strcat(int2str(Percent_Sample),'p_test_list.txt'),'w');
    % Test Subject testing subset
    filename = strcat(int2str(Percent_Sample), 'p_T_test_', 'TS_', int2str(T_list(1)), '.h5');
    fprintf(fileID, '%s\n', strcat('./', filename));
    hdf5write(filename,'/data', T_test.data, '/label',[T_test.label; T_test.headpose], '/data_weights', ones(1, size(T_test.data, 4)));
    %hdf5write(filename,'/data', T_test.data, '/label',[T_test.label; T_test.headpose], '/data_weights', ones(1, size(T_test.data, 4)), '/hp_weights', ones(1, size(T_test.data, 4))) 

    fclose(fileID); 
    fprintf('--- Done with %d ---\n', Percent_Sample);
end
