module MLMC
export mlmc_test_short,mlmc,MLMCResult,MLMCResult_test,statistics, plot_test
using DocOpt
    using Gadfly # package for plotting
    using DataFrames

type MLMCResult
    eps::Float64
    estimate::Float64
    tictoc::Float64
    nums::Array{Float64,1} # would like this to be int
    suml::Array{Float64,2}
    L::Int64
    alpha::Float64
    beta::Float64
end

type MLMCResult_test
    alpha::Float64
    beta::Float64
    gamma::Float64
    kurtosis::Array{Float64,1}
    variance_l::Array{Float64,1}
    mean_l::Array{Float64,1}
    cost_l::Array{Float64,1}
    L::Int64
end


function mlmc(id,eps,mlmc_l, alpha_0=0,beta_0=0,gamma=0,Lmin=4,Lmax=32,N0=1000)

    doc= """ Follows Giles 2008 """
    srand(id);
    if (Lmin<2)
        error("error: needs Lmin >= 2");
    end

    if (Lmax<Lmin)
        error("error: needs Lmax >= Lmin");
    end

    if (N0<=0 || eps<=0 || gamma <= 0)
        error("error: needs N>0, eps>0, gamma>0 \n");
    end

    alpha = alpha_0;
    beta  = beta_0;
    M    = 2.0;
    dNl  = zeros(Lmax+1);
    Nl   = zeros(Lmax+1); # should be int
    dNl[1:(Lmin+1)] = N0;
    suml = zeros(3, Lmax+1)
    L = Lmin;
    # assigns the proportion of the error between Variance and Bias
    theta = 0.25;
    tic()
    while sum(dNl) > 0
        #update sample sums
        for l=0:L
            if dNl[l+1] > 0
#                 @show (ceil(Int64,dNl[l+1]),l);
                start_time=time()
                samples     =  mlmc_l(ceil(Int64,dNl[l+1]),l);
                suml[3,l+1]+= time()-start_time
                Nl[l+1]     = Nl[l+1] +length(samples);
                suml[1,l+1] = suml[1,l+1] + sum(samples);
                suml[2,l+1] = suml[2,l+1] + sum(samples.^2);
            end
        end

        #compute absolute average and variance


        mean_l = abs(vec(suml[1,1:(L+1)])./Nl[1:(L+1)]);
        Vars_l = max(0, vec(suml[2,1:(L+1)])./Nl[1:(L+1)] - mean_l.^2);

        #fix to cope with possible zero values for ml and Vl
        #(can happen in some applications when there are few samples)

        for l = 3:(L+1)
            mean_l[l] = max(mean_l[l], 0.5*mean_l[l-1]/2.0^alpha);
            Vars_l[l] = max(Vars_l[l], 0.5*Vars_l[l-1]/2.0^beta);
        end

        if alpha_0==0
            pa = linreg(vec(float([2:(L+1);])) ,vec(log2(mean_l[2:(L+1)])));
            alpha = -pa[2];
        end
        if beta_0==0
            pb = linreg(vec(float([2:(L+1);])) ,vec(log2(Vars_l[2:(L+1)])));
            beta = -pb[2];
        end

        # Cost of the level. Should be estimated with regression
        Cl  = 2.^(gamma*(0:L));

        #set optimal number of additional samples
        Ns   = zeros(L+1);

        Ns  = ceil(Integer, sqrt(Vars_l./Cl) * sum(sqrt(Vars_l.*Cl))/ ((1-theta)*eps^2) );
        dNl = max(zeros(L+1), Ns-Nl[1:(L+1)]);

        #if (almost) converged, estimate remaining error and decide
        #whether a new level is required

        if maximum(dNl./Nl[1:(L+1)]) < 0.01
            rem = mean_l[L+1] / (2^alpha - 1);
            if rem > sqrt(theta)*eps
                if (L==Lmax)
                    println("*** failed to achieve weak convergence *** ")
                else
                    L       = L+1;
                    Vars_l = push!(Vars_l,Vars_l[L] / 2^beta);
                    Nl[L+1] = 0;
                    Cl  = 2.^(gamma*(0:L));
                    Ns   = zeros(L+1);
                    Ns  = ceil( sqrt(Vars_l./Cl) * sum(sqrt(Vars_l.*Cl)) / ((1-theta)*eps^2) );
                    dNl = max(0, Ns-Nl[1:(L+1)]);
                end
            end
        end
  end
# finally, evaluate multilevel estimator
        tictoc=toq()
    P = sum(vec(suml[1,1:(L+1)])./Nl[1:(L+1)]);

    return MLMCResult(eps,P,tictoc, Nl,suml, L, alpha, beta)
end

function statistics(res::MLMCResult)
    levels=res.L
    nums=res.nums[1:levels]
    means=vec(res.suml[1,1:levels])./nums
    vars= vec(res.suml[2,1:levels])./nums -means.^2
    (means, vars)
end

function mlmc_test_short(mlmc_l,N,L)

    del1 = zeros(L+1);
    var1 = zeros(L+1);
    kur1 = zeros(L+1);
    cost = zeros(L+1);

    for l = 0:L
        tic();
        samples     = mlmc_l(N,l);
        s4   = sum(samples.^4)/N;
        s3   = sum(samples.^3)/N;
        s2   = sum(samples.^2)/N;
        s1   = sum(samples   )/N;
        cost[l+1] = toq();

        if (l==0)
            kurt = 0.0;
        else
            kurt = ( s4 - 4*s3*s1 + 6*s2*s1^2 - 3*s1^4 ) / (s2-s1^2)^2;
        end

        del1[l+1] = s1;
        var1[l+1] = s2-s1^2;
        kur1[l+1] = kurt;
    end

    if ( kur1[L+1] > 100.0 )
        println("\n WARNING: kurtosis on finest level = $(kur1[L+1])\n",);
    end

    L1 = ceil(Integer, 0.4*L)+1;
    L2 = L+1;
    X = float(collect(L1:L2));
    pa = linreg(X,log2(abs(del1[L1:L2])));  alpha = -pa[2];
    pb = linreg(X,log2(abs(var1[L1:L2])));  beta  = -pb[2];
    gamma = log2(cost[L+1]/cost[L]);

#    println("\n******************************************************\n");
#    println("*** Linear regression estimates of MLMC parameters ***\n");
#    println("******************************************************\n");
#    println("\n alpha = $(alpha)  (exponent for MLMC weak convergence)\n");
#    println(" beta  = $(beta)  (exponent for MLMC variance) \n");
#    println(" gamma = $(gamma)  (exponent for MLMC cost) \n");
    return MLMCResult_test(alpha, beta, gamma, kur1, var1, del1, cost, L)

end

function plot_test(res::MLMCResult_test)
    df1 = DataFrame(x=0:res.L, y = vec(res.variance_l) , label="Variance")
    df2 = DataFrame(x=0:res.L, y = vec(10.^(-(0:res.L))), label="First order decay")
    df3 = DataFrame(x=0:res.L, y = vec(abs(res.mean_l)), label="Mean")
    dfp1 = vcat(df1, df2, df3);
    #dfp2 = vcat(df3, df2);

    p1 = plot(dfp1, x="x", y="y", color="label", Scale.y_log,  Guide.xlabel("level"), Guide.ylabel("\log(Variance)"), Guide.title("Variance decay"), Geom.point, Geom.line)
#    p2 = plot(dfp2, x="x", y="y", color="label", Scale.y_log,  Guide.xlabel("level"), Guide.ylabel("\log(Mean)"),     Guide.title("Mean decay"), Geom.point, Geom.line)
#Scale.discrete_color_manual(colorant"blue",colorant"red", colorant"green", colorant"orange",colorant"black"))
    draw(PDF("mlmc_test.pdf", 12inch, 6inch),p1)


end
end
