pl=4;
pd=0.6;

k=1000;

%% this divide raw data into 60-min segment. 

%pre=X(1:300*k,:);
%pre1=X(1:60*k,:);
%pre2=X(60*k+1:120*k,:);
%pre3=X(120*k+1:180*k,:);
%pre4=X(180*k+1:240*k,:);
%pre5=X(240*k+1:300*k,:);
%
%stim=X(300*k+1:600*k,:);
%stim1=X(300*k+1:360*k,:);
%stim2=X(360*k+1:420*k,:);
%stim3=X(420*k+1:480*k,:);
%stim4=X(480*k+1:540*k,:);
%stim5=X(540*k+1:600*k,:);
%
%post=X(600*k+1:900*k,:);
%post1=X(600*k+1:660*k,:);
%post2=X(660*k+1:720*k,:);
%post3=X(720*k+1:780*k,:);
%post4=X(780*k+1:840*k,:);
%post5=X(840*k+1:900*k,:);

%%  All R-R intervals and its positions are found. They have to be checked whether detecting is done correctly or not.

[pks, locs]=findpeaks(X(:,2)*(-1), X(:,1), 'MinPeakHeight', pl, 'MinPeakDistance', pd);

rr=locs(2:end) - locs(1:end-1);
pst=(locs(1:end-1)+locs(2:end)) / 2;

figure(1)
hold on
plot(X(:,1), X(:,2))
plot(locs, pks*(-1), 'ro')
hold off

rr_all=rr;
pst_all=pst;