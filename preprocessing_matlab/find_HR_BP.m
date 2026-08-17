%data=readmatrix('%filename%');
X=data(1:900000,1:3);
%% Heart Rate

find_rri;
HR=60./rr_all;
HR_pst=pst_all;
%% Blood Pressure

BP=X(:,1:2:3);
[sBP, sBP_pst, dBP, dBP_pst, mBP, pp, mBP_pst] = find_BP(BP);

figure
hold on
plot(X(:,1), X(:,3))
plot(sBP_pst, sBP*0.01, 'ro')
plot(dBP_pst, dBP*0.01, 'go')
hold off
