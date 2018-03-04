function [pis,thetas,dict] = naive_bayes_learn(train_data, train_label)
% naive bayes classifier (binary)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%
% Output:
%  pis: the probabilities for each of the classes
%  thetas: conditional probabilities 
%  dict: list of unique terms 
%
% CSCI 576 2014 Fall, Homework 1
% modified so only binary classification

% initialize stuff
sz = size(train_data);
D = sz(2);

cats = [0 1]; %binary categories
pis = zeros(1,2);
% theta(c,k) = P(feature_k=1|y=c), 1-theta(c,k) = P(feature_k=0|y=c), 

dict = cell(1,D);
thetas = cell(2,D);
for k = 1 : D
    dict{k} = unique(train_data(:,k));
    thetas{1,k} = zeros(length(dict{k}),1);
    thetas{2,k} = zeros(length(dict{k}),1);
end

% compute pi_c, theta_c,k
for c = 1:2
    tempData = train_data(train_label==cats(c),:);
    pis(c) = length(tempData);
    tmpThetas = thetas(c,:);
    for d = 1 : D
        flen = length(dict{d});
        feature_count = zeros(flen,1);
        vals = tempData(:,d);
        words = dict{d};
        parfor k = 1 : flen
            feature_count(k) = length(nonzeros(strcmp(vals,words{k})));
        end
        tmpThetas{d} = feature_count;
    end
    thetas(c,:) = tmpThetas;
end
% 
% tempData = train_data(train_label==cats(2),:);
% pis(2) = length(tempData)/N;
% for d = 1 : D
%     flen = length(dict{d});
%     feature_count = zeros(flen,1);
%     for k = 1 : flen
%         feature_count(k) = length(nonzeros(strcmp(tempData(:,d),dict{d}{k})));
%     end
%     thetas{2,d} = feature_count/length(tempData);
% end

% thetas(thetas==0) = 1e-5; %this shouldn't apply for now...

% for n = 1 : N 
%     p = 1;
%     x = train_data(n,:);
%     for d = 1 : D
%         p=p*thetas{1,d}{x{d};
%     end
% end

% % predict and calculate accuracies
% for n = 1 : N 
%     sample = train_data(n,:);
%     mle = -Inf;
%     for c = 1 : C
%         % P(y=c)*P(feature_1=x_1|y=c)*....
%         temp = log(pis(c)) + log(thetas(c,:))*sample' + log(1-thetas(c,:))*~sample';
%         if temp > mle 
%             mle = temp;
%             train_predLabel(n) = cats(c);
%         end
%     end
% end
% 
% train_logloss = sum(train_predLabel == train_label)/N;
% 
% for m = 1 : M 
%     sample = new_data(m,:);
%     mle = -Inf;
%     for c = 1 : C
%         temp = log(pis(c)) + log(thetas(c,:))*sample' + log(1-thetas(c,:))*~sample';
%         if temp > mle 
%             mle = temp;
%             new_predLabel(m) = cats(c);
%         end
%     end
% end
% new_logloss = sum(new_predLabel == new_label)/M;