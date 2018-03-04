cd /auto/rcf-proj2/dos/jianqizh/Avazu
%generate a permutation list;
%nobs = size(train_label,1);
%permute = rand(1,nobs); 
%perm_idx = floor(permute*100)+1;

%save('perm_idx.mat','perm_idx');

load train;
load test;
load dict.mat full_dicts;
%load perm_idx;

r=0.01;

counts = zeros(size(full_dicts));

for d = 1:length(full_dicts)
    counts(d) = length(full_dicts{d});
end

N = length(train_data); Ntest = round(0.9*N);
D = length(counts);

z = cell(D,1);
w = cell(D,1);

for d = 1 : D
	w{d} = zeros(counts(d),1);
end

perm = randperm(size(train_data,1));
valIdx = perm(1:Ntest);
testIdx = perm((Ntest+1),N);

t=cputime;
for n = valIdx
	if (mod(n,10000) == 0)
		disp(['Running block ' num2str(n)]);
	end

	y=train_label(n);
	x = train_data(n,:);

	weights = 0;
	for d = 1:D
	weights = weights + w{d}(x(d));
	end

	yh = (weights>0);

	if (y~=yh)
		for d = 1:D
		w{d}(x(d)) = w{d}(x(d)) + r*(y-yh);
		end
	end
end
cputime-t



%make prediction
N2 = length(testIdx);
y_pred = zeros(N2,1);
y=train_label(testIdx);

ct=0
for n2 = testIdx
	if (mod(n2,10000) == 0)
		disp(['Predicting block ' num2str(n)]);
	end

	x = train_data(n2,:);

	weights = 0;
	for d = 1:D
	weights = weights + w{d}(x(d));
	end
	if weights==0 y_pred(n2)=0.5;
	else y_pred(ct) = (weights>0);
	end
end
	
epss=0.001; %arbitrary value, may be model tuning parameter  
y_pred=min(max(y_pred,epss),1-epss);  
LogLoss=-mean(y.*log(y_pred)+(1-y).*log(1-y_pred))

