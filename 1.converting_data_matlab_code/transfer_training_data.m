fid = fopen('train.csv');
%fid = fopen('train_10k.csv');
headers = strsplit(fgetl(fid),',');

n_block = 5000000;
n_all = 40428968-1; %47686352-1;
numIts = ceil(n_all/n_block);

N = n_all;
D = 22;
train_data = zeros(N,D);
train_label = zeros(N,1);
category_label = cell(1, D);
for k = 1 : numIts
    tic
    data = textscan(fid,['%*s' repmat('%s',1,23)], n_block, 'delimiter',',','CollectOutput',true); %skip the app_id
    data = data{1};
    if k == numIts
        ind = (k-1)*n_block+1 : n_all;
    else
        ind = (k-1)*n_block+1: k*n_block;
    end
    train_label(ind) = str2double(data(:,1));
    disp(['Transfering ' num2str(k) 'th block of ' num2str(n_block) ' lines']);
    
    % keep hour information via trimming off date
    feature_cell = data(:,2);
    hour_cell = cell(length(ind),1);
    for time_i = 1 : length(feature_cell)
        tmp_str = feature_cell{time_i};
        hour_cell{time_i} = tmp_str(7:8);
    end
    data(:,2) = hour_cell;
    
    for i = 2 : D+1
        feature_cell = data(:,i);
        [feature_value, new_category_label] = transfer_category_feature(feature_cell, category_label{i-1});
        train_data(ind, i-1) = feature_value;
        category_label{i-1} = new_category_label;
    end
    toc
end
fclose('all');
save('train.mat', 'train_data', 'train_label', 'category_label','-v7.3')

number = zeros(D,1);
for i = 1:D
    number(i) = length(category_label{i});
end
save('train_category_label.mat', 'category_label','-v7.3')