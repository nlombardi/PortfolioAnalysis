m = [1; 2; 3];
C = [1 -1 0; -1 4 -1; 0 -1 9];

p = Portfolio('assetmean', m, 'assetcovar', C, 'lowerbudget', 0, 'upperbudget', 1, 'lowerbound', -1, 'RiskFreeRate', 1.1);
q = Portfolio('assetmean', m, 'assetcovar', C, 'lowerbudget', 1, 'upperbudget', 1, 'lowerbound', -1);

plotFrontier(p);

p = Portfolio(p, 'lowerbudget', 1);

psgt = estimateMaxSharpeRatio(p);
pwgt = estimateFrontier(p);
[prsk, pret] = estimatePortMoments(p, psgt);


hold on
plot(prsk,pret,'*r');

qsgt = estimateMaxSharpeRatio(q);
qwgt = estimateFrontier(q);
[qrsk, qret] = estimatePortMoments(q, qwgt);
plotFrontier(q);

display(psgt);
display(pwgt);
display(prsk);
display(pret);

title('Mean-Variance Efficient Frontier');
