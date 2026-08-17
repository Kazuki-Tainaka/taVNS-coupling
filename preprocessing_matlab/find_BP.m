function [sBP, sBP_pst, dBP, dBP_pst, mBP, pp, mBP_pst] = find_BP(BP)
%BP
%cheack potions

%% systolic BP
pl = 1.1;
pd = 0.7;
[sBP, sBP_pst] = findpeaks(BP(:,2), BP(:,1), 'MinPeakHeight', pl, 'MinPeakDistance', pd);
sBP = sBP*100;

%% distolic BP
pl = -1.1;
pd = 0.7;
[dBP, dBP_pst] = findpeaks(-1.*BP(:,2), BP(:,1), 'MinPeakHeight', pl, 'MinPeakDistance', pd);
dBP = dBP.*-100;

%% calculate
pp = sBP - dBP;
mBP = pp /3 +dBP;
mBP_pst = (sBP_pst + dBP_pst) /2;
