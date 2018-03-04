function logLoss = calcLogLoss(y_pred, y)
    epss=0.001; %arbitrary value, may be model tuning parameter  
    y_pred_soft=min(max(y_pred,epss),1-epss);  
    logLoss = -mean(y.*log(y_pred_soft)+(1-y).*log(1-y_pred_soft));
    