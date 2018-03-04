function [p, logloss] = ftlr_full_crossvalidation(train_data, train_label, test_data, test_label, counts, alpha, beta, L1, L2)
%Learning models
N = length(train_data);
D = length(counts);

eta = cell(D,1);
z = cell(D,1);

for d = 1 : D
    eta{d} = zeros(counts(d),1);
    z{d} = zeros(counts(d),1);
end

for n = 1 : N
    if (mod(n,10000) == 0)
        disp(['Running block ' num2str(n)]);
    end
    % select a random data point
    x = train_data(n,:);
    y = train_label(n);
    % first predict
    weights = zeros(D,1);
    tmpz = zeros(D,1);
    tmpeta = zeros(D,1);
    for d = 1 : D
        tmpz(d) = z{d}(x(d));
        if tmpz(d) < 0
            sign = -1;
        else
            sign = 1;
        end
        if sign*tmpz(d) <= L1
            weights(d) = 0;
        else
            tmpeta(d) = eta{d}(x(d));
            weights(d) = (sign * L1 - tmpz(d)) / ((beta + sqrt(tmpeta(d))) / alpha + L2);
        end
    end
    wTx = sum(weights);
    p = 1/ (1 + exp(-max(min(wTx,35),-35)));
    % then update
    
    g = p-y;
    sig = sqrt(tmpeta+g*g) - sqrt(tmpeta);
    tmpz = tmpz + g - sig.*weights;
    tmpeta = tmpeta + g*g;
    for d = 1 : D
        z{d}(x(d)) = tmpz(d);
        eta{d}(x(d)) = tmpeta(d);
    end
end
%save ftlr_trained_params z eta

% predict for test data
p = zeros(size(test_data,1),1);
%predict
for n = 1 : size(test_data,1)
	x = test_data(n,:);
	% y = val_label(n);
	% first predict
	weights = zeros(D,1);
	tmpz = zeros(D,1);
	tmpeta = zeros(D,1);
    for d = 1 : D
		if x(d) > counts(d)
			continue
		end
        tmpz(d) = z{d}(x(d));
        if tmpz(d) < 0
            sign = -1;
        else
            sign = 1;
        end
        if sign*tmpz(d) <= L1
            weights(d) = 0;
        else
            tmpeta(d) = eta{d}(x(d));
            weights(d) = (sign * L1 - tmpz(d)) / ((beta + sqrt(tmpeta(d))) / alpha + L2);
        end
    end
    wTx = sum(weights);
    p(n) = 1/ (1 + exp(-max(min(wTx,35),-35)));
end

 logloss = calcLogLoss(p, test_label);
