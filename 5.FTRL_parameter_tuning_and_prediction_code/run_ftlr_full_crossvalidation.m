%load train_100k.mat
%load test.mat
load train.mat
load dict.mat full_dicts

counts = zeros(size(full_dicts));
for d = 1 : length(full_dicts)
    counts(d) = length(full_dicts{d});
end

% all_train_data = train_data;
% all_train_label = train_label;
% index = randperm(size(all_train_data,1),100000);
% train_data = all_train_data(index,:);
% train_label = all_train_label(index,:);
% save train_100k train_data train_label

% train_data = all_train_data;
% train_label = all_train_label;

train_n = length(train_label);
group_n = ceil(train_n / 5); %200
LogLoss = zeros(6,1);
t = zeros(6,1);

%shuffle the data order
index = randperm(size(train_data,1));
suffle_train_data = train_data(index,:);
suffle_train_label = train_label(index);
clear train_data train_label index;

% L1, L2 are penalties to enforce sparsity: 10^{-1:1}
% alpha, beta are learning parameters that control the learning rate:
% 0.9:0.1:1.1
% alpha = 0.1;
% beta = 0.1;
% L1 = 1;
% L2 = 1;

for k = 1:1:5
    display(k)
    valid_index = (k-1)*group_n+1 : k*group_n;
    if k == 5
        valid_index = (k-1)*group_n+1 : train_n;
    end
    train_index = setdiff(1:train_n, valid_index);
    cross_train_data = suffle_train_data(train_index,:);
    cross_train_label = suffle_train_label(train_index,:);
    cross_valid_data = suffle_train_data(valid_index,:);
    cross_valid_label = suffle_train_label(valid_index,:);
    clear train_index valid index;
    tic;
    [p, logloss] = ftlr_full_crossvalidation(cross_train_data, cross_train_label, cross_valid_data, cross_valid_label, counts, alpha, beta, L1, L2);
    t(k) = toc;
    display(t(k))
    display(logloss)
    LogLoss(k) = logloss;
end
t(6) = mean(t(1:5));
LogLoss(6) = mean(LogLoss(1:5));
output_table = table(t, LogLoss);
writetable(output_table, strcat('alpha', num2str(alpha),'_beta', num2str(beta), '_L1', num2str(L1), '_L2', num2str(L2), '_performance_lr_5cross.csv'));
