function p = naive_bayes_test(pis,thetas,dict,data,click)

if nargin > 4
    flag = 1;
else
    flag = 0;
end
sz = size(data);
N = sz(1);
D = sz(2);
p = zeros(N,1);
thetas_1 = thetas(1,:);
thetas_2 = thetas(2,:);
% tic
for n = 1 : N
%     if mod(n,5000)==0
%         disp(n)
% 	toc;
% 	tic;
%     end
    features = data(n,:);
    p_click = pis(2);
    p_nclick = pis(1);
    tmp = p_nclick/p_click;
    for d = 1 : D
        p_1 = thetas_1{d}(strcmp(features{d},dict{d}));
        if isempty(p_1) || p_1 < 1e-7 
            p_1 = 1e-7;
        end
        p_2 = thetas_2{d}(strcmp(features{d},dict{d}));
        if isempty(p_2) || p_1 < 1e-7
            p_2 = 1e-7;
        end
        tmp = tmp*p_1/p_2;
    end
    p(n) = 1/(1+tmp);
end

%accuracy(for fun)
if flag
    pred_click = p;
    pred_click(pred_click<0.5) = 0;
    pred_click(pred_click>0.5) = 1;
    disp(['Accuracy is: ' num2str(mean(pred_click==click))]);
end
