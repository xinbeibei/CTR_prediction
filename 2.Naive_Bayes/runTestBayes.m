fclose('all');
% fid = fopen('train_rev2.csv');
fid = fopen('test_rev2.csv');
numIts = ceil(4769402/100000);
%i=randi(47686351*179,1,1);
% i = 698020345;
%i=7731710292;
final_pis = all_pis/sum(all_pis);
% for k = 1 : length(all_thetas)
%   all_thetas{1,k} = all_thetas{1,k}/all_pis(1);
%   all_thetas{2,k} = all_thetas{2,k}/all_pis(2);
% end

% final_dicts = all_dicts;
% [~,idx] = sort(final_thetas{1,10},'descend');
% final_thetas{1,10} = final_thetas{1,10}(idx(1:2000));
% final_thetas{2,10} = final_thetas{2,10}(idx(1:2000));
% final_dicts{1,10} = final_dicts{1,10}(idx(1:2000));
% [~,idx] = sort(final_thetas{1,11},'descend');
% final_thetas{1,11} = final_thetas{1,11};
% final_thetas{2,11} = final_thetas{2,11}(idx(1:2000));
% final_dicts{1,11} = final_dicts{1,11}(idx(1:2000));
%

% fseek(fid,i,'bof');
fgetl(fid);
%ys = zeros(4769401,1);
p = zeros(4769401,1);
count = 1;
for k = 1 : numIts
    tic
    disp([num2str(k) 'th part of 100000']);
    % make sure the line below is correct for training vs testing
    data = textscan(fid,['%*s' repmat('%s',1,25)],100000, 'delimiter',',','CollectOutput',true);
    data = data{1};
    %y = str2double(data(:,1));
%     data(:,1) = [];
%     data(:,[10 11]) = [];
    p(count:count+length(data)-1) = naive_bayes_test(final_pis,final_thetas,final_dicts,data);
 %   ys(count:count+length(data)-1) = y;
    count = count+length(data);
    toc
end
fclose('all');

epss=0.001; %arbitrary value, may be model tuning parameter
% y_pred=min(max(p,epss),1-epss);
% LogLoss=-mean(ys.*log(y_pred)+(1-ys).*log(1-y_pred));