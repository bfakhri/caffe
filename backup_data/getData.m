function Ret_Data = getData(source_dirs, divider)
% getData returns 1/divider of the data for the participants in the list 
% source_dirs


for count_d=1:length(source_dirs)
    path = dir(source_dirs{count_d});
    path = path(3:end);
    files = {path.name};

    total_num=0;
    for num_f=1:length(files)
        readname = [source_dirs{count_d}, files{num_f}];
        temp = load(readname);

        total_num = total_num;
        %total_num = total_num+length(temp.errors);

    end

    Data=[];
    Data.data = zeros(60,36,1, total_num*2);
    Data.label = zeros(2, total_num*2);
    Data.headpose = zeros(2, total_num*2);
    Data.confidence = zeros(1, total_num*2);

    index = 0;

    for num_f=1:length(files)
        readname = [source_dirs{count_d}, files{num_f}];
        temp = load(readname);

        num_data = length(temp.filenames(:,1));     
        for num_i=1:num_data
            % for left
            index = index+1;
            img = temp.data.left.image(num_i, :,:);
            img = reshape(img, 36,60);
            Data.data(:, :, 1, index) = img'; % filp the image

            Lable_left = temp.data.left.gaze(num_i, :)';
            theta = asin((-1)*Lable_left(2));
            phi = atan2((-1)*Lable_left(1), (-1)*Lable_left(3));
            Data.label(:,index) = [theta; phi];

            headpose = temp.data.left.pose(num_i, :);
            M = rodrigues(headpose);
            Zv = M(:,3);
            theta = asin(Zv(2));
            phi = atan2(Zv(1), Zv(3));
            Data.headpose(:,index) = [theta;phi];         

            % for right
            index = index+1;
            img = temp.data.right.image(num_i, :,:);
            img = reshape(img, 36,60);
            Data.data(:, :, 1, index) = double(flip(img, 2))'; % filp the image

            Lable_right = temp.data.right.gaze(num_i,:)';
            theta = asin((-1)*Lable_right(2));
            phi = atan2((-1)*Lable_right(1), (-1)*Lable_right(3));
            Data.label(:,index) = [theta; (-1)*phi];% flip the direction

            headpose = temp.data.right.pose(num_i, :); 
            M = rodrigues(headpose);
            Zv = M(:,3);
            theta = asin(Zv(2));
            phi = atan2(Zv(1), Zv(3));
            Data.headpose(:,index) = [theta; (-1)*phi]; % flip the direction
        end
        %fprintf('Read %d of %d Files in %s\n', num_f, length(files), char(source_dirs(count_d))); 
    end
    fprintf('Read in File %s\n', char(source_dirs(count_d))); 

    Data.data = Data.data/255;      %normalize
    Data.data = single(Data.data);  % must be single data, because caffe want float type
    Data.label = single(Data.label);
    Data.headpose = single(Data.headpose);
    
    % Divides by the dividing factor
    %size(Data.data)
    Data.data = Data.data(:,:,:,1:divider:end);
    Data.label = Data.label(:,1:divider:end);
    Data.headpose = Data.headpose(:,1:divider:end);
    
    if(count_d > 1)
        Ret_Data.data = cat(4, Ret_Data.data, Data.data);
        Ret_Data.label = cat(2, Ret_Data.label, Data.label);
        Ret_Data.headpose = cat(2, Ret_Data.headpose, Data.headpose);
    else
        Ret_Data.data = Data.data;
        Ret_Data.label = Data.label;
        Ret_Data.headpose = Data.headpose;
    end
end