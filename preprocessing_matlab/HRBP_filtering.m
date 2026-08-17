function [zeroHR, zeroBP] = HRBP_filtering(HR_pst, HR, mBP_pst, mBP)
    Rp=10; %passband ripple
    
    steps=30/64;
    
    xx=0:steps:900;
    
    %cubic spline interpolation
    splineHR=spline(HR_pst, HR, xx);
    splineBP=spline(mBP_pst, mBP, xx);
    
    %filtering
    [b,a]=cheby1(3, Rp, [0.08 0.12], 'bandpass');
    
    filterHR=filter(b, a, splineHR);
    filterBP=filter(b, a, splineBP);
    
    zeroHR=filterHR - mean(filterHR);
    zeroBP=(filterBP - mean(filterBP))*100;
end