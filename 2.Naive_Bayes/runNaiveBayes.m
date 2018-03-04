%read the data in

fid = fopen('train_rev2.csv');
headers = strsplit(fgetl(fid),',');
numIts = ceil(47686352/100000);
%numIts = 2;
all_pis = zeros(1,2);
all_thetas = cell(2,25);
all_dicts = cell(1,25);
for k = 1 : numIts
    tic
    data = textscan(fid,['%*s' repmat('%s',1,26)],100000, 'delimiter',',','CollectOutput',true);
    data = data{1};
    click = str2double(data(:,1));
    data(:,1) = [];
    disp(['Analyzing ' num2str(k) 'th block of 100000']);
    [pis,thetas,dicts] = naive_bayes_learn(data, click);
    all_pis = all_pis + pis;
    [all_thetas,all_dicts] = joinData(all_thetas,all_dicts,thetas,dicts);
    toc
end
fclose('all');


% join them together
for d = 1 : 25
    all_thetas{1,d} = all_thetas{1,d}/all_pis(1);
    all_thetas{2,d} = all_thetas{2,d}/all_pis(2);
end
final_pis = all_pis/sum(all_pis);
%testing

fseek(fid,randi(4786352,1,1)*179,'bof');
fgetl(fid);
data = textscan(fid,['%*s' repmat('%s',1,26)],100000, 'delimiter',',','CollectOutput',true);
data = data{1};
click = str2double(data(:,1));
data(:,1) = [];
p = naive_bayes_test(final_pis,all_thetas,all_dicts,data,click);
