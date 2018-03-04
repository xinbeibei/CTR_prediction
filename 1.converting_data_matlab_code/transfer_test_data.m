%fid = fopen('train_rev2.csv');
load('train_category_label.mat') %category_label
fid = fopen('test.csv');
headers = strsplit(fgetl(fid),',');

n_block = 500000;
n_all = 4577465-1;
numIts = ceil(n_all/n_block);

N = n_all;
D = 22;
test_data = zeros(N,D);

for k = 1 : numIts
    tic
    data = textscan(fid,['%*s' repmat('%s',1,22)], n_block, 'delimiter',',','CollectOutput',true); %skip the app_id
    data = data{1};
    if k == numIts
        ind = (k-1)*n_block+1 : n_all;
    else
        ind = (k-1)*n_block+1: k*n_block;
    end
    disp(['Transfering ' num2str(k) 'th block of ' num2str(n_block) ' lines']);
    
    feature_cell = data(:,1);
    hour_cell = cell(length(ind),1);
    for time_i = 1 : length(feature_cell)
        tmp_str = feature_cell{time_i};
        hour_cell{time_i} = tmp_str(7:8);
    end
    
    data(:,1) = hour_cell;
    
    for i = 1 : D
        feature_cell = data(:,i);
        [feature_value, new_category_label] = transfer_category_feature(feature_cell, category_label{i});
        test_data(ind, i) = feature_value;
        category_label{i} = new_category_label;
    end
    toc
end
fclose('all');

save('test.mat', 'test_data', 'category_label','-v7.3')

number = zeros(D,1);
for i = 1:D
    number(i) = length(category_label{i});
end
save('train_test_category_label.mat', 'category_label','-v7.3')
