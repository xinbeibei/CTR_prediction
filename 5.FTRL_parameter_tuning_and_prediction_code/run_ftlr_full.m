load train.mat
load test.mat
load dict.mat full_dicts

counts = zeros(size(full_dicts));
for d = 1 : length(full_dicts)
    counts(d) = length(full_dicts{d});
end

alpha = 0.8;
beta = 0.8;
L1 = 0.0001;
L2 = 0.0001;

p = ftlr_full(train_data,train_label,test_data,counts, alpha, beta, L1, L2);

%write table
fid = fopen('test.id');
id = textscan(fid, '%s');
id = id{1};
fclose(fid);
id = id(2:end);
click = p;
output_table = table(id, click);
writetable(output_table, strcat('alpha', num2str(alpha),'_beta', num2str(beta), '_L1', num2str(L1), '_L2', num2str(L2), '_prediction_logisticRegression.csv'));
