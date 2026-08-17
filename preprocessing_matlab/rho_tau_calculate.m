function [rhomax, taumax, rhomax_pst] = rho_tau_calculate(zeroHR, zeroBP)
    step = 30/64;

    rhomax=zeros(1,60*14/step);
    taumax=zeros(1,60*14/step);

    d = 60 / step;

    w = hann(d * 2) .';

    for i = d + 1 : 1 : 14 * d
        sepHR = zeroHR(i - d : i + d - 1);          %divide HR and BP by two minutes
        sepBP = zeroBP(i - d : i + d - 1);
    
        sepHR = sepHR .* w;
        sepBP = sepBP .* w;                     %apply hanning filter
    
        top = xcorr(-1 * sepHR, sepBP);          %calculate rho
    
        rHR = xcorr(sepHR);
        rBP = xcorr(sepBP);
    
        l = length(rHR);
    
        bottom = sqrt(rHR(round(l/2)) * rBP(round(l/2)));
    
        rho = top / bottom;
    
        [tmp_rhomax, tmp_taumax] = max(rho);     %find rhomax and taumax

        rhomax(1,i) = tmp_rhomax;                      %rhomax matrix
        taumax(1,i) = tmp_taumax;                      %taumax matrix
    end
   
    rhomax(1:128)=[];
    taumax(1:128)=[];

    rhomax_pst=60+step:step:840;
end

