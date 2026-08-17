function makedata(HR, HR_pst, sBP, sBP_pst, dBP, dBP_pst, rhomax)
    step = 30/64;
    time = 0:step:900;

    data=zeros(1921, 5);

    data(:,1)=time;

    tmp=spline(HR_pst, HR, time);
    data(:,2)=tmp;

    tmp=spline(sBP_pst, sBP, time);
    data(:,3)=tmp;

    tmp=spline(dBP_pst, dBP, time);
    data(:,4)=tmp;

    %rhomax(1)=[];
    zz=zeros(128, 1);
    tmp=cat(1, 0, zz, rhomax.', zz);
    data(:,5)=tmp;

    raw_HR=cat(2, HR_pst, HR);
    %rhomax_f=cat(2, rhomax_f, rhomax);

    raw_sBP=cat(2, sBP_pst, sBP);
    raw_dBP=cat(2, dBP_pst, dBP);

    writematrix(data, 'data_n.csv');
    writematrix(raw_HR, 'raw_HR_n.csv');
    writematrix(raw_sBP, 'raw_sBP_n.csv');
    writematrix(raw_dBP, 'raw_dBP_n.csv');
    %writetable(rhomax_f, 'rhomax_f.csv');

end