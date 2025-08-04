__precompile__

# using ProgressMeter
using ProgressBars
using LinearAlgebra
using FastGaussQuadrature, LegendrePolynomials
using LinearSolve
using LowRankApprox
using PyCall
using Roots
using SparseArrays
using IntervalSets
using JLD2

include("SNSystem.jl")
include("utils.jl")
include("stencils.jl")
include("StaggeredGrid.jl")
include("boundayconds.jl")

mutable struct SolverMarshak{T<:AbstractFloat}
    # Spatial grid of cell vertices
    x::Array{T};
    xMid::Array{T};
    y::Array{T};
    yMid::Array{T};

    # Solver settings
    settings::Settings;

    ## Spatial discretisation
    SG::StaggeredGrid; # All the information about the staggered grid
    # Stencil matrices
    stencil::Stencils;

    ## Angular discretisation
    sn::SNSystem;

    # Physical parameters
    SigmaA_mac::SparseMatrixCSC{T, Int64};
    SigmaT_mic::SparseMatrixCSC{T, Int64};
    SigmaT_mic_inv::SparseMatrixCSC{T, Int64};

    density::SparseMatrixCSC{T, Int64};
    
    T::DataType;

    # Constructor
    function SolverMarshak(settings)
        T = Float64;

        settings.x = T.(settings.x);
        settings.xMid = T.(settings.xMid);
        settings.y = T.(settings.y);
        settings.yMid = T.(settings.yMid);

        x = settings.x;
        xMid = settings.xMid;
        y = settings.y;
        yMid = settings.yMid;

        sn = SNSystem(settings,T);
        SetupSNprojected2D(sn);

        # Stencil matrices for spatial discretisation
        SG = StaggeredGrid(settings,T);
        stencil = Stencils(s,T);

        # Setting up the material coefficients
        N_MicroGrid = SG.N_MicroGrid;
        N_MacroGrid = SG.N_MacroGrid;
        SigmaA_mac = sparse([],[],[],N_MacroGrid,N_MacroGrid);
        SigmaT_mic = sparse([],[],[],N_MicroGrid,N_MicroGrid);
        SigmaT_mic_inv = sparse([],[],[],N_MicroGrid,N_MicroGrid);
        density = sparse([],[],[],N_MacroGrid,N_MacroGrid);
        
        # Setup temporal discretisation
        Omegax_min = abs(maximum(sn.points[:,1]));
        Omegay_min = abs(maximum(sn.points[:,2]));
        AdvecSpeed = settings.AdvecSpeed;
        if settings.cfltype == "h" # stands for hyperbolic
            settings.dt = settings.cfl*min(settings.epsilon*settings.dx,settings.epsilon*settings.dy)/3/AdvecSpeed; 
        elseif settings.cfltype == "p"
            settings.dt = settings.cfl*min(settings.sigmaT*settings.dx^2/4,settings.sigmaT*settings.dy^2/4)/3/AdvecSpeed; 
        elseif settings.cfltype == "m"
            settings.dt = settings.cfl*min(settings.epsilon*settings.dx + settings.sigmaT*settings.dx^2/4,settings.epsilon*settings.dy + settings.sigmaT*settings.dy^2/4)/3/AdvecSpeed;
        else
            println("Please enter a valid cfltype")
        end

        new{T}(T.(x),T.(xMid),T.(y),T.(yMid),settings,SG,stencil,sn,SigmaA_mac,SigmaT_mic,SigmaT_mic_inv,density,T)
    end
end

py"""
import numpy
def qr(A):
    return numpy.linalg.qr(A)
"""

function SetupMaterialConstants(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    N_MicroGrid = obj.SG.N_MicroGrid;
    if obj.settings.problem == "Hohlraum"
        II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals1 = zeros(T,N_MicroGrid);
        
        for k = 1:N_MicroGrid
            xi,yi = obj.SG.micro_grid[k];
            II[k] = k; J[k] = k; 
            if 0.0 <= xi <= 1.0 && (0.0 <= yi <= 0.05 || 0.95 <= yi <= 1.0)
                vals1[k] = 100; 
            elseif 0.95 <= xi <= 1.0 && 0.0 <= yi <= 1.0
                vals1[k] = 100; 
            elseif 0.0 <= xi <= 0.05 && 0.25 <= yi <= 0.75
                vals1[k] = 100; 
            elseif 0.25 <= xi <= 0.75 && 0.25 <= yi <= 0.75
                vals1[k] = 100; 
            else
                vals1[k] = 0.0; 
            end
        end

        SigmaT_mic = sparse(II,J,T.(vals1),N_MicroGrid,N_MicroGrid);
        SigmaT_mic_inv = sparse(II,J,T.(vals1),N_MicroGrid,N_MicroGrid);

        II = zeros(Int,N_MacroGrid); J = zeros(Int,N_MacroGrid); vals = zeros(T,N_MacroGrid);
        vals_density = zeros(T,N_MacroGrid); 
        for k = 1:N_MacroGrid
            xi,yi = obj.SG.macro_grid[k];
            II[k] = k; J[k] = k; 
            if  0.0 <= xi <= 1.0 && (0.0 <= yi <= 0.05 || 0.95 <= yi <= 1.0)
                vals[k] = 100; 
                vals_density[k] = 1.0;
            elseif 0.95 <= xi <= 1.0 && 0.0 <= yi <= 1.0
                vals[k] = 100;
                vals_density[k] = 1.0; 
            elseif 0.0 <= xi <= 0.05 && 0.25 <= yi <= 0.75
                vals[k] = 100; 
                vals_density[k] = 1.0;
            elseif  0.25 <= xi <= 0.75 && 0.25 <= yi <= 0.75
                vals[k] = 100; 
                vals_density[k] = 1.0; 
            else
                vals[k] = 0; 
                vals_density[k] = 1e94; 
            end
        end
        SigmaA_mac = sparse(II,J,T.(vals),N_MacroGrid,N_MacroGrid);
        density = sparse(II,J,T.(vals_density),N_MacroGrid,N_MacroGrid);
    else
        II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals = zeros(T,N_MicroGrid); vals_inv = zeros(T,N_MicroGrid);
        for k = 1:N_MicroGrid
            II[k] = k; J[k] = k; vals[k] = obj.settings.sigmaA; vals_inv[k] = 1/obj.settings.sigmaA;
        end

        SigmaT_mic = sparse(II,J,T.(vals),N_MicroGrid,N_MicroGrid);
        SigmaT_mic_inv = sparse(II,J,T.(vals_inv),N_MicroGrid,N_MicroGrid);

        II = zeros(Int,N_MacroGrid); J = zeros(Int,N_MacroGrid); vals = zeros(T,N_MacroGrid); 
        for k = 1:N_MacroGrid
            II[k] = k; J[k] = k; vals[k] = obj.settings.sigmaA;
        end
        SigmaA_mac = sparse(II,J,T.(vals),N_MacroGrid,N_MacroGrid);
        density = sparse(I(N_MacroGrid));
    end
    obj.SigmaA_mac = SigmaA_mac;
    obj.SigmaT_mic = SigmaT_mic;
    obj.SigmaT_mic_inv = SigmaT_mic_inv;
    obj.density = density;
end

function SetupIC(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    N_MicroGrid = obj.SG.N_MicroGrid;
    N_MacroGrid = obj.SG.N_MacroGrid;
    TNv = obj.sn.TNv;
    
    h = zeros(T,N_MacroGrid);
    g = zeros(T,N_MicroGrid,TNv);
    Temp = zeros(T,N_MacroGrid);

    ghost_micro = union(obj.SG.ghostidx_XMidY,obj.SG.ghostidx_XYMid);
    if obj.settings.problem == "Gaussian" || obj.settings.problem == "MarshakWave"||  obj.settings.problem == "Hohlraum" 
        Temp = ICTemp(obj.settings,obj.SG.macro_grid,obj.SG.boundaryidx_mac_L,obj.SG.ghostidx_mac_L);
        parden,scalflux = ICParDen(obj.settings,obj.SG.micro_grid,obj.SG.macro_grid,obj.sn.points,ghost_micro,Temp);
    end
    h = ICh(obj.settings,scalflux,Temp);
    g = ICg(obj.settings,parden,scalflux);

    return h,g,Temp;
end

function ComputeEnergy(obj::SolverMarshak,BT,Temp,h,g) 
    AdvecSpeed = obj.settings.AdvecSpeed;
    epsilon = obj.settings.epsilon;
    c_nu = obj.settings.c_nu;
    aRad = obj.settings.aRad;
    dzeta = obj.settings.dx*obj.settings.dy;
    if g isa CuArray
        M = CuSparseMatrixCSC(obj.sn.M);
    else
        M = obj.sn.M;
    end

    sum1 = 0.0;
    sum2 = 0.0;
    sum3 = 0.0;

    for i = 1:obj.SG.N_MacroGrid
        sum1 += (1/AdvecSpeed*BT[i] + epsilon^2/AdvecSpeed * h[i])^2;
        sum2 += aRad*c_nu/4/pi^2 * (Temp[i])^5; 
    end

    sum3 = (epsilon/AdvecSpeed)^2*tr(g*M*g');

    energy = sum1*dzeta + 1/2/pi*sum3*dzeta + 2/5*sum2*dzeta;
    return energy;
end

function ComputeEnergy(obj::SolverMarshak,BT,Temp,h,X,S,V) 
    AdvecSpeed = obj.settings.AdvecSpeed;
    epsilon = obj.settings.epsilon;
    c_nu = obj.settings.c_nu;
    aRad = obj.settings.aRad;
    dzeta = obj.settings.dx*obj.settings.dy;
    if X isa CuArray
        M = CuSparseMatrixCSC(obj.sn.M);
    else
        M = obj.sn.M;
    end

    sum1 = 0.0;
    sum2 = 0.0;
    sum3 = 0.0;

    for i = 1:obj.SG.N_MacroGrid
        sum1 += (1/AdvecSpeed*BT[i] + epsilon^2/AdvecSpeed * h[i])^2;
        sum2 += aRad*c_nu/4/pi^2 * (Temp[i])^5; 
    end
    
    sum3 = (epsilon/AdvecSpeed)^2*tr(X*S*V'*M*V*S'*X');

    energy = sum1*dzeta + 1/2/pi*sum3*dzeta + 2/5*sum2*dzeta;
    return energy;
end

function ComputeMass(obj::SolverMarshak,BT,T,h)
    AdvecSpeed = obj.settings.AdvecSpeed;
    c_nu = obj.settings.c_nu;
    epsilon = obj.settings.epsilon;
    sum1 = 0.0;
    
    for i = 1:obj.SG.N_MacroGrid
        if i ∉ obj.SG.ghostidx_XY
            sum1 += (2*pi/AdvecSpeed*(BT[i] + epsilon^2 * h[i]) + c_nu*T[i])*obj.settings.dx*obj.settings.dy;
        end
    end
    return sum1;
end

function BundleSolution(obj::SolverMarshak,h,g,Temp, ranks, energy, mass)
    Dict1 = Dict("h" => h, "g" => g, "Temp" => Temp, "phi" => obj.settings.aRad*obj.settings.AdvecSpeed/2/pi.*Temp.^4 + obj.settings.epsilon^2 .*h, "ranks" => ranks, "energy" => energy, "mass" => mass);
    return Dict1;
end

function BundleSolutionDL(obj::SolverMarshak,Temp)
    Dict1 = Dict("Temp" => Temp,"phi" => obj.settings.aRad*obj.settings.AdvecSpeed/2/pi.*Temp.^4);
    return Dict1;
end

function save_solution(problem,name,val,k,t)
    folder = "2DMarshakNL/results/" * problem * "/";
    namefile = folder * name * ".jld2";
    if k == 0
        save(namefile);
        jldopen(namefile, "w") do file
            write(file, "$k", val)  # alternatively, say "@write file A"
        end
    else
        jldopen(namefile, "a+") do file
            write(file, "$k", val)  # alternatively, say "@write file A"
        end
    end
end

function SolveDiffusionLimit(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    dt = obj.settings.dt;

    Nt = Int(round(obj.settings.tEnd/dt));

    NCellsX = obj.settings.NCellsX;
    Nx = obj.settings.Nx;

    aRad = obj.settings.aRad;
    AdvecSpeed = obj.settings.AdvecSpeed;
    c_nu = obj.settings.c_nu;

    SetupMaterialConstants(obj);

    density = obj.density;

    # Store the intial condition
    _,_,Temp = SetupIC(obj);
    Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
    BT = aRad*AdvecSpeed.*Temp.^4 ./2/pi;
    Dox = obj.stencil.Dox;
    Doy = obj.stencil.Doy;
    deltaox = obj.stencil.deltaox;
    deltaoy = obj.stencil.deltaoy;

    Diff = Dox*obj.SigmaT_mic_inv*deltaox .+ Doy*obj.SigmaT_mic_inv*deltaoy;

    Diff_BT = zeros(obj.T, obj.SG.N_MacroGrid);

    N_MacroGrid = obj.SG.N_MacroGrid;
    boundaryidx = obj.SG.boundaryidx_macro;
    ghostidx = obj.SG.ghostidx_XY;

    for k = ProgressBar(1:Nt)
        
        Diff_BT .= Diff*BT;

        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY
                rhsᵢ = density[i,i]*c_nu*Temp[i]  + dt*2*pi*(Diff_BT[i])/3 + 2*pi*BT[i]/AdvecSpeed;
                # println(i, ", ",rhsᵢ)
                fi(x) = density[i,i]*c_nu*x + aRad*x^4 - rhsᵢ;
                Temp[i] = find_zero(fi,Temp[i]);
            end
        end
        Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
        BT .= aRad * AdvecSpeed .* Temp.^4 ./2/pi ;
    end
    sol = BundleSolutionDL(obj,Temp);
    return sol;
end

function SolveFull(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    t = 0.0;
    dt = obj.settings.dt;
    Nt = Int(round(obj.settings.tEnd/dt));

    ϵ = obj.settings.epsilon;

    NCellsX = obj.settings.NCellsX;
    Nx = obj.settings.Nx;
    N_MicroGrid = obj.SG.N_MicroGrid;
    N_MacroGrid = obj.SG.N_MacroGrid;

    aRad = obj.settings.aRad;
    AdvecSpeed = obj.settings.AdvecSpeed;
    cᵥ = obj.settings.c_nu;

    SetupMaterialConstants(obj);

    density = obj.density;
    # Store the intial condition
    h,g,Temp = SetupIC(obj);
    
    g .= BCg(obj.settings,obj.SG,obj.sn,g,h,Temp);
    Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
    h = BCh(obj.settings,obj.SG,obj.sn,h,g,Temp);
    BT = aRad*AdvecSpeed.*Temp.^4 ./2/pi;
  
    # Boundary and ghost cells
    ghostidx_XMidY = obj.SG.ghostidx_XMidY;
    ghostidx_XYMid = obj.SG.ghostidx_XYMid;
    ghostidx_XY = obj.SG.ghostidx_XY
    boundaryidx_micro = obj.SG.boundaryidx_micro;
    boundaryidx_macro = obj.SG.boundaryidx_macro;
    
    II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals = zeros(obj.T,N_MicroGrid);
    for k = 1:N_MicroGrid
        II[k] = k; J[k] = k; vals[k] = 1/(1/AdvecSpeed + dt*obj.SigmaT_mic[k,k]/ϵ^2);
    end
    Beta = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

    L2x = obj.stencil.L2x;
    L2y = obj.stencil.L2y;
    L1x = obj.stencil.L1x;
    L1y = obj.stencil.L1y;
    Dox = obj.stencil.Dox;
    Doy = obj.stencil.Doy;
    deltaox = obj.stencil.deltaox;
    deltaoy = obj.stencil.deltaoy;

    Qx = obj.sn.Qx;
    Qy = obj.sn.Qy;
    AbsQx = obj.sn.AbsQx;
    AbsQy = obj.sn.AbsQy;

    w = obj.sn.w;

    unitvec = ones(obj.sn.TNv);
    AdvecMat = (I(obj.sn.TNv) - 1/2/pi .* obj.sn.w * Transpose(unitvec));

    QxAM_epsi  = Qx*AdvecMat ./ϵ;
    QyAM_epsi = Qy*AdvecMat ./ϵ;
    AbsQxAM_epsi = AbsQx*AdvecMat ./ϵ;
    AbsQyAM_epsi = AbsQy*AdvecMat ./ϵ;

    Qx1_epsi2 = unitvec'Qx ./ϵ^2;
    Qy1_epsi2 = unitvec'Qy ./ϵ^2;
    BTh = zeros(obj.T,obj.SG.N_MacroGrid);

    R1,R2,R3,R4 = zeros(obj.T,obj.SG.N_MicroGrid,obj.sn.TNv),zeros(obj.T,obj.SG.N_MicroGrid,obj.sn.TNv),zeros(obj.T,obj.SG.N_MicroGrid,obj.sn.TNv),zeros(obj.T,obj.SG.N_MicroGrid,obj.sn.TNv);

    Grad_g = zeros(obj.SG.N_MacroGrid);

    energy = zeros(Nt+1);
    mass = zeros(Nt+1);
    energy[1] = ComputeEnergy(obj,BT,Temp,h,g);
    mass[1] = ComputeMass(obj,BT,Temp,h);
    g_BC = zeros(size(g));

    for k = ProgressBar(1:Nt) 
        BTh .= BT .+ ϵ^2 .* h;
        R1 .= L2x*g*QxAM_epsi;
        R2 .= L1x*g*AbsQxAM_epsi;
        R3 .= L2y*g*QyAM_epsi;
        R4 .= L1y*g*AbsQyAM_epsi;
        g .= g./AdvecSpeed .- dt .*(R1 .- R2 .+ R3 .- R4) .- dt .*deltaox*BTh*Qx1_epsi2  .- dt .*deltaoy*BTh*Qy1_epsi2;
        g .= Beta * g;

        g .= BCg(obj.settings,obj.SG,obj.sn,g,h,Temp);

        ## Macro-meso update
        Grad_g .= (Dox * g * Qx * w) .+ (Doy * g * Qy * w);
        
        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY
                σᵢ = obj.SigmaA_mac[i,i];
                αᵢ = 1 + dt*AdvecSpeed*σᵢ/ϵ^2;
                rhsᵢ = density[i,i]*cᵥ*Temp[i] + 2*pi*dt*σᵢ/αᵢ*(h[i] + 1/ϵ^2*BT[i] - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
                f(x) = density[i,i]*cᵥ*x + (dt*aRad*AdvecSpeed*σᵢ/αᵢ/ϵ^2)*x^4 - rhsᵢ;
                Temp[i] = find_zero(f,Temp[i]);
            end
        end

        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY 
                αᵢ = 1 + dt*AdvecSpeed*obj.SigmaA_mac[i,i]/ϵ^2;
                BT1 = aRad*AdvecSpeed/2/pi*(Temp[i])^4;
                h[i] = 1/αᵢ*(h[i] - 1/ϵ^2*(BT1 - BT[i]) - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
            end
        end
        # BC
        Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
         # BC
        h .= BCh(obj.settings,obj.SG,obj.sn,h,g,Temp);


        BT .= aRad*AdvecSpeed.*Temp.^4 ./2/pi;

        energy[k+1] = ComputeEnergy(obj,BT,Temp,h,g);
        mass[k+1] = ComputeMass(obj,BT,Temp,h);

        t = t + dt;
    end
    sol = BundleSolution(obj,h,g,Temp,zeros(Nt), energy, mass);
    return sol;
end

function SolvefrBUG(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    t = 0.0;
    dt = obj.settings.dt;
    Nt = Int(round(obj.settings.tEnd/dt));

    ϵ = obj.settings.epsilon;

    NCellsX = obj.settings.NCellsX;
    Nx = obj.settings.Nx;
    N_MicroGrid = obj.SG.N_MicroGrid;
    N_MacroGrid = obj.SG.N_MacroGrid;
    r = obj.settings.r;

    aRad = obj.settings.aRad;
    AdvecSpeed = obj.settings.AdvecSpeed;
    cᵥ = obj.settings.c_nu;

    SetupMaterialConstants(obj);

    density = obj.density;
    
    # Boundary and ghost cells
    ghostidx_XMidY = obj.SG.ghostidx_XMidY;
    ghostidx_XYMid = obj.SG.ghostidx_XYMid;
    ghostidx_XY = obj.SG.ghostidx_XY
    boundaryidx_micro = obj.SG.boundaryidx_micro;
    boundaryidx_macro = obj.SG.boundaryidx_macro;

    # Storing the stencil and quadrtature matrices
    L2x = obj.stencil.L2x;
    L2y = obj.stencil.L2y;
    L1x = obj.stencil.L1x;
    L1y = obj.stencil.L1y;
    Dox = obj.stencil.Dox;
    Doy = obj.stencil.Doy;
    deltaox = obj.stencil.deltaox;
    deltaoy = obj.stencil.deltaoy;

    Qx = obj.sn.Qx;
    Qy = obj.sn.Qy;
    AbsQx = obj.sn.AbsQx;
    AbsQy = obj.sn.AbsQy;

    w = obj.sn.w;

    # Store the intial condition
    h,g,Temp = SetupIC(obj);
    g .= BCg(obj,g,h,T);
    Temp .= BCTemp(obj,obj.settings.problem,Temp,Nx,Ny);
    BT = aRad*AdvecSpeed.*Temp.^4 ./2/pi;
    h .= BCh(obj.SG,obj.settings.problem,ϵ,h,BT);

    # Initialising the K, L and S matrices
    X,S,V = svd!(g);

    X = Matrix(X[:,1:r]);
    V = Matrix(V[:,1:r]);
    S = diagm(S[1:r]);
    L = zeros(size(V'));
    K = zeros(size(X));

    X .= BCg(obj.SG,obj.settings.problem,X);


    Xnew = zeros(obj.T,size(X));
    Vnew = zeros(obj.T,size(V));

    # Pre- allocating memory to speed up computations
    VQxV = zeros(obj.T,r,r);
    VQyV = zeros(obj.T,r,r);
    VAbsQxV = zeros(obj.T,r,r);
    VAbsQyV = zeros(obj.T,r,r);

    XL2xX = zeros(obj.T,r,r);
    XL2yX = zeros(obj.T,r,r);
    XL1xX = zeros(obj.T,r,r);
    XL1yX = zeros(obj.T,r,r);

    MUp = zeros(obj.T,r,r);
    NUp = zeros(obj.T,r,r);

    unitvec = ones(obj.sn.TNv);
    AdvecMat = (I(obj.sn.TNv) - 1/2/pi .* obj.sn.w * Transpose(unitvec));
    
    II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals = zeros(obj.T,N_MicroGrid);
    for k = 1:N_MicroGrid
        II[k] = k; J[k] = k; vals[k] = 1/(1/AdvecSpeed + dt*obj.SigmaT_mic[k,k]/ϵ^2);
    end
    Beta_K = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);
    Beta = zeros(Float64,r,r);

    QxAM  = Qx*AdvecMat;
    QyAM = Qy*AdvecMat;
    AbsQxAM = AbsQx*AdvecMat;
    AbsQyAM = AbsQy*AdvecMat;
    Qx1 = unitvec'Qx;
    Qy1 = unitvec'Qy;

    # VQxV .= V'*QxAM*V;
    # VQyV .= V'*QyAM*V;
    # VAbsQxV .= V'*AbsQxAM*V;
    # VAbsQyV .= V'*AbsQyAM*V;

    # XL2xX .= X'*L2x*X;
    # XL2yX .= X'*L2y*X;
    # XL1xX .= X'*L1x*X;
    # XL1yX .= X'*L1y*X;

    BTh = zeros(obj.T,obj.SG.N_MacroGrid);
    Grad_g = zeros(obj.T,obj.SG.N_MacroGrid);
    deltaBTh = zeros(obj.T,obj.SG.N_MicroGrid,obj.sn.TNv);

    ranks = zeros(Int64,Nt+1);
    ranks[1] = r;
    check_g = zeros(Nt);
    energy = zeros(Nt+1);
    mass = zeros(Nt);
    energy[1] = ComputeEnergy(obj,BT,Temp,h,g);
    mass[1] = ComputeMass(obj,BT,Temp,h);

    for k = ProgressBar(1:Nt)
        ## Micro update
        BTh .= BT .+ ϵ^2 .* h;
        deltaBTh .= deltaox*BTh*Qx1 .+ deltaoy*BTh*Qy1;
        #K-step
        K .= X*S;

        K = BCg(obj.SG,obj.settings.problem,K);

        VQxV .= V'*QxAM*V;
        VQyV .= V'*QyAM*V;
        VAbsQxV .= V'*AbsQxAM*V;
        VAbsQyV .= V'*AbsQyAM*V;
        
        K .= K./AdvecSpeed .- dt/ϵ .*L2x * K * VQxV .+ dt/ϵ .*L1x * K * VAbsQxV  .- dt/ϵ .*L2y * K * VQyV  .+ dt/ϵ .*L1y * K * VAbsQyV .- dt/ϵ^2 .*deltaBTh*V;
        K .= Beta_K*K;

        Xnew,_ = py"qr"(K);

        Xnew .= BCg(obj.SG,obj.settings.problem,Xnew);

        MUp .= Xnew'*X;

        #L-step
        L .= S*V';

        Beta .= 1/AdvecSpeed .*I(r) .+ dt/ϵ^2 .* X'*obj.SigmaT_mic*X;
        
        XL2xX .= X'*L2x*X;
        XL2yX .= X'*L2y*X;
        XL1xX .= X'*L1x*X;
        XL1yX .= X'*L1y*X;

        L .= L./AdvecSpeed .- dt/ϵ .*XL2xX*L*QxAM .+ dt/ϵ .* XL1xX*L*AbsQxAM .- dt/ϵ .*XL2yX*L*QyAM .+ dt/ϵ .* XL1yX*L*AbsQyAM .- dt/ϵ^2 .*X'*deltaBTh;
        L .= Beta\L;

        Vnew,_ = py"qr"(L');
        
        NUp .= Vnew'*V;

        V .= Vnew;
        X .= Xnew;
        
        #S-step
        S .= MUp*S*(NUp');
        
        Beta .= 1/AdvecSpeed .*I(r) .+ dt/ϵ^2 .* X'*obj.SigmaT_mic*X;

        VQxV .= V'*QxAM*V;
        VQyV .= V'*QyAM*V;
        VAbsQxV .= V'*AbsQxAM*V;
        VAbsQyV .= V'*AbsQyAM*V;

        XL2xX .= X'*L2x*X;
        XL2yX .= X'*L2y*X;
        XL1xX .= X'*L1x*X;
        XL1yX .= X'*L1y*X;


        S .= S./AdvecSpeed .- dt/ϵ .*XL2xX*S*VQxV .+ dt/ϵ .* XL1xX*S*VAbsQxV .- dt/ϵ .*XL2yX*S*VQyV .+ dt/ϵ .* XL1yX*S*VAbsQyV .- dt/ϵ^2 .*X'*deltaBTh*V;
        S .= Beta\S;

        ranks[k+1] = r;
        check_g[k] = norm(X*S*V'*w)/norm(X*S*V');


        # println(check_g[k]);

        ## Macro-meso update
        Grad_g .= ((Dox * X * S) *( V' * Qx * w)) .+ ((Doy * X * S) * (V' * Qy * w));
        for i = 1:N_MacroGrid
            σᵢ = obj.SigmaA_mac[i,i];
            αᵢ = 1 + dt*AdvecSpeed*σᵢ/ϵ^2;
            rhsᵢ = Temp[i] + 2*pi*dt*σᵢ/αᵢ/density[i,i]/cᵥ*(h[i] + 1/ϵ^2*BT[i] - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
            f(x) = x + (dt*aRad*AdvecSpeed*σᵢ/αᵢ/density[i,i]/cᵥ/ϵ^2)*x^4 - rhsᵢ;
            Temp[i] = find_zero(f,Temp[i]);
        end
        Temp .= BCTemp(obj,obj.settings.problem,Temp,Nx,Ny);

        for i = 1:N_MacroGrid
            αᵢ = 1 + dt*AdvecSpeed*obj.SigmaA_mac[i,i]/ϵ^2;
            BT1 = aRad*AdvecSpeed/2/pi*(Temp[i])^4;
            h[i] = 1/αᵢ*(h[i] - 1/ϵ^2*(BT1 - BT[i]) - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
        end
        # BC
        h .= BCh(obj,h,g);
        Temp .= BCTemp(obj,obj.settings.problem,Temp,Nx,Ny);
        
        energy[k+1] = ComputeEnergy(obj,BT,Temp,h,X*S*V');
        mass[k+1] = ComputeMass(obj,BT,Temp,h);

        t = t + dt;
    end
    sol = BundleSolution(obj,h,X*S*V',Temp,ranks, energy, mass);
    return sol;
end


function SolveAugBUG(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    t = 0.0;
    dt = obj.settings.dt;
    Nt = Int(round(obj.settings.tEnd/dt));

    ϵ = obj.settings.epsilon;

    NCellsX = obj.settings.NCellsX;
    Nx = obj.settings.Nx;
    N_MicroGrid = obj.SG.N_MicroGrid;
    N_MacroGrid = obj.SG.N_MacroGrid;
    r = obj.settings.r;

    aRad = obj.settings.aRad;
    AdvecSpeed = obj.settings.AdvecSpeed;
    cᵥ = obj.settings.c_nu;

    SetupMaterialConstants(obj);

    density = obj.density;
    
    # Boundary and ghost cells
    ghostidx_XMidY = obj.SG.ghostidx_XMidY;
    ghostidx_XYMid = obj.SG.ghostidx_XYMid;
    ghostidx_XY = obj.SG.ghostidx_XY
    boundaryidx_micro = obj.SG.boundaryidx_micro;
    boundaryidx_macro = obj.SG.boundaryidx_macro;

    boundary_ghost_idx = sort(union(obj.SG.ghostidx_XMidY,obj.SG.ghostidx_XYMid,obj.SG.boundaryidx_micro));

    # Storing the stencil and quadrtature matrices
    L2x = obj.stencil.L2x;
    L2y = obj.stencil.L2y;
    L1x = obj.stencil.L1x;
    L1y = obj.stencil.L1y;
    Dox = obj.stencil.Dox;
    Doy = obj.stencil.Doy;
    deltaox = obj.stencil.deltaox;
    deltaoy = obj.stencil.deltaoy;

    Qx = obj.sn.Qx;
    Qy = obj.sn.Qy;
    AbsQx = obj.sn.AbsQx;
    AbsQy = obj.sn.AbsQy;

    w = obj.sn.w;

    # Store the intial condition
    h,g,Temp = SetupIC(obj);
    Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
    # g .= BCg(obj.settings,obj.SG,obj.sn,g,h,T);
    # h .= BCh(obj.settings,obj.SG,obj.sn,h,g);
    BT = aRad*AdvecSpeed.*Temp.^4 ./2/pi;
   

    # Initialising the K, L and S matrices
    X,S,V = svd!(g);

    X = Matrix(X[:,1:r]);
    V = Matrix(V[:,1:r]);
    S = diagm(S[1:r]);

    # fBC = zeros(size(g));
    # XBC = BCg(obj.SG,obj.settings.problem,fBC) * V;
    # X[obj.SG.boundaryidx_micro,:] .= XBC[obj.SG.boundaryidx_micro,:]

    Xnew = zeros(obj.T,size(X));
    Vnew = zeros(obj.T,size(V));

    # Pre- allocating memory to speed up computations
    VQxV = zeros(obj.T,r,r);
    VQyV = zeros(obj.T,r,r);
    VAbsQxV = zeros(obj.T,r,r);
    VAbsQyV = zeros(obj.T,r,r);

    XL2xX = zeros(obj.T,r,r);
    XL2yX = zeros(obj.T,r,r);
    XL1xX = zeros(obj.T,r,r);
    XL1yX = zeros(obj.T,r,r);

    MUp = zeros(obj.T,2*r,r);
    NUp = zeros(obj.T,2*r,r);

    unitvec = ones(obj.sn.TNv);
    AdvecMat = (I(obj.sn.TNv) - 1/2/pi .* obj.sn.w * Transpose(unitvec));
    
    II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals = zeros(obj.T,N_MicroGrid);
    for k = 1:N_MicroGrid
        II[k] = k; J[k] = k; vals[k] = 1/(1/AdvecSpeed + dt*obj.SigmaT_mic[k,k]/ϵ^2);
    end
    Beta_K = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

    QxAM  = Qx*AdvecMat;
    QyAM = Qy*AdvecMat;
    AbsQxAM = AbsQx*AdvecMat;
    AbsQyAM = AbsQy*AdvecMat;
    Qx1 = unitvec'Qx;
    Qy1 = unitvec'Qy;
    e1 = zeros(obj.T,obj.SG.N_MicroGrid);
    e1[1] = 1.0;

    BTh = zeros(obj.T,obj.SG.N_MacroGrid);
    BT_augx = zeros(obj.T,obj.SG.N_MicroGrid);
    BT_augy = zeros(obj.T,obj.SG.N_MicroGrid);
    Grad_g = zeros(obj.T,obj.SG.N_MacroGrid);

    ranks = zeros(Int64,Nt+1);
    ranks[1] = r;

    energy = zeros(Nt+1);
    mass = zeros(Nt+1);
    energy[1] = ComputeEnergy(obj,BT,Temp,h,g);
    mass[1] = ComputeMass(obj,BT,Temp,h);

    for k = ProgressBar(1:Nt)
        BT_augx .= obj.SigmaT_mic_inv*deltaox*BT;
        BT_augy .= obj.SigmaT_mic_inv*deltaoy*BT;

        ## Micro update
        BTh .= BT .+ ϵ^2 .* h;

        ## Including Boundary condition into the problem
        f = zeros(size(g));
        f .= BCg(obj.settings,obj.SG,obj.sn,g,h,Temp);
        fBC = f[obj.SG.boundaryidx_micro,:];
        # Compute singular values of the BC
        PBC,sigBC,QBC = svd(fBC);
        rb = 1;
        for i = 1:r
            if sigBC[i] < 10^-10
                rb = i-1;
                break;
            end
        end
        QBC = QBC[:,1:rb];
        # Create random basis for x
        PBC = randn(Float64,(N_MicroGrid,rb));
        X_wBC,Sx = py"qr"([X PBC]); # With boundary conditions
        V_wBC,Sv = py"qr"([V QBC]); # With boundary conditions
        S_wBC = zeros(r+rb,r+rb);
        S_wBC[1:r,1:r] .= S; 
        r = r + rb;
        S = Sx*S_wBC*Sv';
        X = X_wBC;
        V = V_wBC;


        # XBC = f * V;
        # X[obj.SG.ghostidx_XMidY,:] = XBC[obj.SG.ghostidx_XMidY,:];
        # X[obj.SG.ghostidx_XYMid,:] = XBC[obj.SG.ghostidx_XYMid,:];

        if k == 1
            h .= BCh(obj.settings,obj.SG,obj.sn,h,X*S*V',Temp);
        end
        

        #K-step
        K = X*S;

        VQxV = V'*QxAM*V;
        VQyV = V'*QyAM*V;
        VAbsQxV = V'*AbsQxAM*V;
        VAbsQyV = V'*AbsQyAM*V;

        K .= K./AdvecSpeed .- dt/ϵ .*L2x * K * VQxV .+ dt/ϵ .*L1x * K * VAbsQxV  .- dt/ϵ .*L2y * K * VQyV  .+ dt/ϵ .*L1y * K * VAbsQyV .- dt/ϵ^2 .*deltaox*BTh*Qx1*V .- dt/ϵ^2 .*deltaoy*BTh*Qy1*V;
        K .= Beta_K*K;

        if obj.settings.problem == "Hohlraum"
            Xnew,_ = py"qr"([K X]);
            tot = 2*r ;
        else
            Xnew,_ = py"qr"([BT_augx BT_augy K X]);
            tot = 2*r + 2;
        end

        MUp = Xnew'*X;

        #L-step
        L = S*V';

        Beta_L = 1/AdvecSpeed .*I(r) + dt/ϵ^2 .* X'*obj.SigmaT_mic*X;

        XL2xX = X'*L2x*X;
        XL2yX = X'*L2y*X;
        XL1xX = X'*L1x*X;
        XL1yX = X'*L1y*X;

        L .= L./AdvecSpeed .- dt/ϵ .*XL2xX*L*QxAM .+ dt/ϵ .* XL1xX*L*AbsQxAM .- dt/ϵ .*XL2yX*L*QyAM .+ dt/ϵ .* XL1yX*L*AbsQyAM .- dt/ϵ^2 .*X'*deltaox*BTh*Qx1 .- dt/ϵ^2 .*X'*deltaoy*BTh*Qy1;
        L .= Beta_L\L;

        
        if obj.settings.problem == "Hohlraum"
            V1,_ = py"qr"([w L' V]);
            Vnew = V1[:,2:end];
        else
            V1,_ = py"qr"([w Qx*unitvec Qy*unitvec L' V]); # w
            Vnew = V1[:,2:end];
        end
        
        NUp = Vnew'*V;

        V = Vnew;
        X = Xnew;

        #S-step
        S = MUp*S*(NUp');
        
        Beta_S = 1/AdvecSpeed .*I(tot) + dt/ϵ^2 .* X'*obj.SigmaT_mic*X;

        VQxV = V'*QxAM*V;
        VQyV = V'*QyAM*V;
        VAbsQxV = V'*AbsQxAM*V;
        VAbsQyV = V'*AbsQyAM*V;

        XL2xX = X'*L2x*X;
        XL2yX = X'*L2y*X;
        XL1xX = X'*L1x*X;
        XL1yX = X'*L1y*X;


        S .= S./AdvecSpeed .- dt/ϵ .*XL2xX*S*VQxV .+ dt/ϵ .* XL1xX*S*VAbsQxV .- dt/ϵ .*XL2yX*S*VQyV .+ dt/ϵ .* XL1yX*S*VAbsQyV .- dt/ϵ^2 .*X'*deltaox*BTh*Qx1*V .- dt/ϵ^2 .*X'*deltaoy*BTh*Qy1*V;
        S .= Beta_S\S;

        #Conservative truncation
        Khat = X * S;
        m = 2; # Number of basis vectors to left unchanged 
        Khat_ap, Khat_rem = Khat[:,1:m], Khat[:,m+1:end]; # Splitting Khat into basis required for Ap and remaining vectors
        Vap, Vrem = V[:,1:m], V[:,m+1:end]; # Splitting Khat into basis required for Ap and remaining vectors

        Xhrem, Shrem = py"qr"(Khat_rem);
        # Xhrem = Matrix(Xhrem);

        U, sigma, W = svd(Shrem);
        U = Matrix(U);
        W = Matrix(W);

        rmax = -1
        tmp = 0.0;
        tol = obj.settings.epsAdapt * norm(sigma);

        # Truncating the rank
        for i = 1:tot-m
            tmp = sqrt(sum(sigma[i:tot-m].^2));
            if tmp < tol
                rmax = i
                # println(rmax,tmp,tol);
                break
            end
        end
        rmaxTotal = Int(round(max(N_MicroGrid, obj.sn.TNv)/2));

        rmax = min(rmax,rmaxTotal);
        rmax = max(rmax,1);

        if rmax == -1
            rmax = rmaxTotal;
        end

        Uhat = U[:,1:rmax];
        What = W[:,1:rmax];
        sigma_hat = diagm(sigma[1:rmax]);

        W1 = Vrem * What;
        Xrem = Xhrem * Uhat;

        V = [Vap W1];
        Xap, Sap = py"qr"(Khat_ap);
        X, R2 = py"qr"([Xap Xrem]);

        S = zeros(rmax+m,rmax+m);
        S[1:m,1:m] = Sap;
        S[m+1:end,m+1:end] = sigma_hat;
        S .= R2 * S;
        r = rmax+m;
        ranks[k+1] = r;

        ## Macro-meso update
        Grad_g .= ((Dox * X * S )* (V' * Qx * w)) .+ ((Doy * X * S) * (V' * Qy * w));
        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY
                σᵢ = obj.SigmaA_mac[i,i];
                αᵢ = 1 + dt*AdvecSpeed*σᵢ/ϵ^2;
                rhsᵢ = density[i,i]*cᵥ*Temp[i] + 2*pi*dt*σᵢ/αᵢ*(h[i] + 1/ϵ^2*BT[i] - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
                f(x) = density[i,i]*cᵥ*x + (dt*aRad*AdvecSpeed*σᵢ/αᵢ/ϵ^2)*x^4 - rhsᵢ;
                Temp[i] = find_zero(f,Temp[i]);
            end
        end
        

        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY
                αᵢ = 1 + dt*AdvecSpeed*obj.SigmaA_mac[i,i]/ϵ^2;
                BT1 = aRad*AdvecSpeed/2/pi*(Temp[i])^4;
                h[i] = 1/αᵢ*(h[i] - 1/ϵ^2*(BT1 - BT[i]) - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
            end
        end
        # BC
        h .= BCh(obj.settings,obj.SG,obj.sn,h,X*S*V',Temp);
        Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);

        BT .= aRad*AdvecSpeed.*Temp.^4 ./2/pi;

        energy[k+1] = ComputeEnergy(obj,BT,Temp,h,X*S*V');
        mass[k+1] = ComputeMass(obj,BT,Temp,h);
        t = t + dt;
    end
    sol = BundleSolution(obj,h,X*S*V',Temp,ranks, energy, mass);
    return sol;
end

function SolveParBUG(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    t = 0.0;
    dt = obj.settings.dt;
    Nt = Int(round(obj.settings.tEnd/dt));

    ϵ = obj.settings.epsilon;

    NCellsX = obj.settings.NCellsX;
    Nx = obj.settings.Nx;
    N_MicroGrid = obj.SG.N_MicroGrid;
    N_MacroGrid = obj.SG.N_MacroGrid;
    r = obj.settings.r;

    aRad = obj.settings.aRad;
    AdvecSpeed = obj.settings.AdvecSpeed;
    cᵥ = obj.settings.c_nu;

    SetupMaterialConstants(obj);
    
    density = obj.density;
    # Boundary and ghost cells
    ghostidx_XMidY = obj.SG.ghostidx_XMidY;
    ghostidx_XYMid = obj.SG.ghostidx_XYMid;
    ghostidx_XY = obj.SG.ghostidx_XY
    boundaryidx_micro = obj.SG.boundaryidx_micro;
    boundaryidx_macro = obj.SG.boundaryidx_macro;

    boundary_ghost_idx = sort(union(obj.SG.ghostidx_XMidY,obj.SG.ghostidx_XYMid,obj.SG.boundaryidx_micro));
    ghostidx_micro = union(obj.SG.ghostidx_XMidY,obj.SG.ghostidx_XYMid)

    # Storing the stencil and quadrtature matrices
    L2x = obj.stencil.L2x;
    L2y = obj.stencil.L2y;
    L1x = obj.stencil.L1x;
    L1y = obj.stencil.L1y;
    Dox = obj.stencil.Dox;
    Doy = obj.stencil.Doy;
    deltaox = obj.stencil.deltaox;
    deltaoy = obj.stencil.deltaoy;

    Qx = obj.sn.Qx;
    Qy = obj.sn.Qy;
    AbsQx = obj.sn.AbsQx;
    AbsQy = obj.sn.AbsQy;

    w = obj.sn.w;

    # Store the intial condition
    h,g,Temp = SetupIC(obj);
    g =  BCg(obj.settings,obj.SG,obj.sn,g,h,Temp);

    # Initialising the K, L and S matrices
    X,S,V = svd!(g);

    X = Matrix(X[:,1:r]);
    V = Matrix(V[:,1:r]);
    S = diagm(S[1:r]);

    Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
    h .= BCh(obj.settings,obj.SG,obj.sn,h,X*S*V',Temp);
    BT = aRad*AdvecSpeed.*Temp.^4 ./2/pi;

    Xnew = zeros(obj.T,size(X));
    Vnew = zeros(obj.T,size(V));
    Xtmp = zeros(obj.T,size(X));
    Vtmp = zeros(obj.T,size(V));

    # Pre- allocating memory to speed up computations
    VQxV = zeros(obj.T,r,r);
    VQyV = zeros(obj.T,r,r);
    VAbsQxV = zeros(obj.T,r,r);
    VAbsQyV = zeros(obj.T,r,r);

    XL2xX = zeros(obj.T,r,r);
    XL2yX = zeros(obj.T,r,r);
    XL1xX = zeros(obj.T,r,r);
    XL1yX = zeros(obj.T,r,r);

    unitvec = ones(obj.sn.TNv);
    AdvecMat = (I(obj.sn.TNv) - 1/2/pi .* obj.sn.w * Transpose(unitvec));
    
    II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals = zeros(obj.T,N_MicroGrid);
    for k = 1:N_MicroGrid
        II[k] = k; J[k] = k; vals[k] = 1/(1/AdvecSpeed + dt*obj.SigmaT_mic[k,k]/ϵ^2);
    end
    Beta_K = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

    QxAM  = Qx*AdvecMat;
    QyAM = Qy*AdvecMat;
    AbsQxAM = AbsQx*AdvecMat;
    AbsQyAM = AbsQy*AdvecMat;
    Qx1 = unitvec'Qx;
    Qy1 = unitvec'Qy;

    BTh = zeros(obj.T,obj.SG.N_MacroGrid);
    BT_augx = zeros(obj.T,obj.SG.N_MicroGrid);
    BT_augy = zeros(obj.T,obj.SG.N_MicroGrid);
    Grad_g = zeros(obj.T,obj.SG.N_MacroGrid);

    ranks = zeros(Int64,Nt+1);
    ranks[1] = r;

    energy = zeros(Nt+1);
    mass = zeros(Nt+1);
    energy[1] = ComputeEnergy(obj,BT,Temp,h,X,S,V);
    mass[1] = ComputeMass(obj,BT,Temp,h);

    # save_solution(obj.settings.problem,"parallel_solver",Temp,0,0);
    material_idx = findall(x->x==100,obj.SigmaA_mac);

    for k = ProgressBar(1:Nt)
        
        BT_augx .= obj.SigmaT_mic_inv*deltaox*BT;
        BT_augy .= obj.SigmaT_mic_inv*deltaoy*BT;
        ## Micro update
        BTh .= BT .+ ϵ^2 .* h;

        # Augmenting the basis functions
        if obj.settings.problem == "Hohlraum" 

            Xaug = X;
            Vaug1,_ = py"qr"([w V]);
            Vaug = Vaug1[:,2:end];

            Saug = S * (V' *Vaug); #

            tot = r;
            totS = 2*tot;
        else
            Xaug,_ = py"qr"([BT_augx BT_augy X]);
            Vaug1,_ = py"qr"([w Qx*unitvec Qy*unitvec V]); # w 
            Vaug = Vaug1[:,2:end];

            Saug = (Xaug' * X) * S * (V' *Vaug);
            
            tot = r+2;
            totS = 2*tot;
            
        end        

        #K-step
        K = Xaug*Saug;
        
        VQxV = Vaug'*QxAM*Vaug;
        VQyV = Vaug'*QyAM*Vaug;
        VAbsQxV = Vaug'*AbsQxAM*Vaug;
        VAbsQyV = Vaug'*AbsQyAM*Vaug;

        K .= K./AdvecSpeed .- dt/ϵ .*L2x * K * VQxV .+ dt/ϵ .*L1x * K * VAbsQxV  .- dt/ϵ .*L2y * K * VQyV  .+ dt/ϵ .*L1y * K * VAbsQyV .- dt/ϵ^2 .*deltaox*BTh*Qx1*Vaug .- dt/ϵ^2 .*deltaoy*BTh*Qy1*Vaug;
        K .= Beta_K*K;
        
        Xtmp,_ = py"qr"([Xaug K]);

        X1Tilde = Xtmp[:,(tot+1):end];

        #L-step
        L = Saug*Vaug';

        Beta = 1/AdvecSpeed .*I(tot) .+ dt/ϵ^2 .* Xaug'*obj.SigmaT_mic*Xaug;

        XL2xX = Xaug'*L2x*Xaug;
        XL2yX = Xaug'*L2y*Xaug;
        XL1xX = Xaug'*L1x*Xaug;
        XL1yX = Xaug'*L1y*Xaug;

        L .= L./AdvecSpeed .- dt/ϵ .*XL2xX*L*QxAM .+ dt/ϵ .* XL1xX*L*AbsQxAM .- dt/ϵ .*XL2yX*L*QyAM .+ dt/ϵ .* XL1yX*L*AbsQyAM .- dt/ϵ^2 .*Xaug'*deltaox*BTh*Qx1 .- dt/ϵ^2 .*Xaug'*deltaoy*BTh*Qy1;
        
        L .= Beta\L;
        Vtmp,_ = py"qr"([w Vaug L']);
        V1Tilde = Vtmp[:,(tot+2):end];
  
        #S-step

        Saug .= Saug./AdvecSpeed .- dt/ϵ .*XL2xX*Saug*VQxV .+ dt/ϵ .* XL1xX*Saug*VAbsQxV .- dt/ϵ .*XL2yX*Saug*VQyV .+ dt/ϵ .* XL1yX*Saug*VAbsQyV .- dt/ϵ^2 .*Xaug'*deltaox*BTh*Qx1*Vaug .- dt/ϵ^2 .*Xaug'*deltaoy*BTh*Qy1*Vaug;
        Saug .= Beta\Saug;

        S = zeros(totS,totS);
        S[1:(tot),1:(tot)] .= Saug;
        S[(tot+1):end,1:(tot)] .= X1Tilde' * K;
        S[1:(tot),(tot+1):end] .= L * V1Tilde;

        X = [Xaug X1Tilde];
        V = [Vaug V1Tilde];
        

        #Conservative truncation
        Khat = X * S;
        m = 2; # Number of basis vectors to left unchanged 
        Khat_ap, Khat_rem = Khat[:,1:m], Khat[:,m+1:end]; # Splitting Khat into basis required for Ap and remaining vectors
        Vap, Vrem = V[:,1:m], V[:,m+1:end]; # Splitting Khat into basis required for Ap and remaining vectors

        Xhrem, Shrem = qr(Khat_rem);
        Xhrem = Matrix(Xhrem);

        U, sigma, W = svd(Shrem);
        U = Matrix(U);
        W = Matrix(W);

        rmax = -1
        tmp = 0.0;
        tol = obj.settings.epsAdapt * norm(sigma);

        # Truncating the rank
        for i = 1:totS-m
            tmp = sqrt(sum(sigma[i:totS-m].^2));
            if tmp < tol
                rmax = i
                break
            end
        end
        rmaxTotal = Int(round(max(N_MicroGrid, obj.sn.TNv)/2));

        rmax = min(rmax,rmaxTotal);
        rmax = max(rmax,1);

        if rmax == -1
            rmax = rmaxTotal;
        end

        Uhat = U[:,1:rmax]; 
        What = W[:,1:rmax];
        sigma_hat = diagm(sigma[1:rmax]);

        W1 = Vrem * What;
        Xrem = Xhrem * Uhat;

        V = [Vap W1];
        Xap, Sap = py"qr"(Khat_ap);
        X, R2 = py"qr"([Xap Xrem]);

        S = zeros(rmax+m,rmax+m);
        S[1:m,1:m] = Sap;
        S[m+1:end,m+1:end] = sigma_hat;
        S .= R2 * S;
        r = rmax+m;
        ranks[k+1] = r;

        # ## Including Boundary condition into the problem
        X,S,V = BCg_lr(obj.settings,obj.SG,obj.sn,X,S,V,h,Temp);


        ## Macro-meso update
        Grad_g .= ((Dox * X * S )* (V' * Qx * w)) .+ ((Doy * X * S) * (V' * Qy * w));
        
        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY
                σᵢ = obj.SigmaA_mac[i,i];
                αᵢ = 1 + dt*AdvecSpeed*σᵢ/ϵ^2;
                rhsᵢ = density[i,i]*cᵥ*Temp[i] + 2*pi*dt*σᵢ/αᵢ*(h[i] + 1/ϵ^2*BT[i] - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
                f(x) = density[i,i]*cᵥ*x + (dt*aRad*AdvecSpeed*σᵢ/αᵢ/ϵ^2)*x^4 - rhsᵢ;
                Temp[i] = find_zero(f,Temp[i]);
            end
        end

        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY 
                αᵢ = 1 + dt*AdvecSpeed*obj.SigmaA_mac[i,i]/ϵ^2;
                BT1 = aRad*AdvecSpeed/2/pi*(Temp[i])^4;
                h[i] = 1/αᵢ*(h[i] - 1/ϵ^2*(BT1 - BT[i]) - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
            end
        end
        # BC
        Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
        h .= BCh(obj.settings,obj.SG,obj.sn,h,X*S*V',Temp);
        BT .= aRad*AdvecSpeed.*Temp.^4 ./2/pi;

        energy[k+1] = ComputeEnergy(obj,BT,Temp,h,X,S,V);
        mass[k+1] = ComputeMass(obj,BT,Temp,h);

        
        t = t + dt;
        # if k%10 == 0
        #     save_solution(obj.settings.problem,"parallel_solver",Temp,k,t);
        # end
    end
    sol = BundleSolution(obj,h,X*S*V',Temp,ranks, energy, mass);
    return sol;
end

function SolveParBUGasync(obj::SolverMarshak{T}) where {T<:AbstractFloat}
    t = 0.0;
    dt = obj.settings.dt;
    Nt = Int(round(obj.settings.tEnd/dt));

    ϵ = obj.settings.epsilon;

    NCellsX = obj.settings.NCellsX;
    Nx = obj.settings.Nx;
    N_MicroGrid = obj.SG.N_MicroGrid;
    N_MacroGrid = obj.SG.N_MacroGrid;
    r = obj.settings.r;

    aRad = obj.settings.aRad;
    AdvecSpeed = obj.settings.AdvecSpeed;
    cᵥ = obj.settings.c_nu;

    SetupMaterialConstants(obj);
    
    density = obj.density;
    # Boundary and ghost cells
    ghostidx_XMidY = obj.SG.ghostidx_XMidY;
    ghostidx_XYMid = obj.SG.ghostidx_XYMid;
    ghostidx_XY = obj.SG.ghostidx_XY
    boundaryidx_micro = obj.SG.boundaryidx_micro;
    boundaryidx_macro = obj.SG.boundaryidx_macro;

    boundary_ghost_idx = sort(union(obj.SG.ghostidx_XMidY,obj.SG.ghostidx_XYMid,obj.SG.boundaryidx_micro));
    ghostidx_micro = union(obj.SG.ghostidx_XMidY,obj.SG.ghostidx_XYMid)

    # Storing the stencil and quadrtature matrices
    L2x = obj.stencil.L2x;
    L2y = obj.stencil.L2y;
    L1x = obj.stencil.L1x;
    L1y = obj.stencil.L1y;
    Dox = obj.stencil.Dox;
    Doy = obj.stencil.Doy;
    deltaox = obj.stencil.deltaox;
    deltaoy = obj.stencil.deltaoy;

    Qx = obj.sn.Qx;
    Qy = obj.sn.Qy;
    AbsQx = obj.sn.AbsQx;
    AbsQy = obj.sn.AbsQy;

    w = obj.sn.w;

    # Store the intial condition
    h,g,Temp = SetupIC(obj);
    g =  BCg(obj.settings,obj.SG,obj.sn,g,h,Temp);

    # Initialising the K, L and S matrices
    X,S,V = svd!(g);

    X = Matrix(X[:,1:r]);
    V = Matrix(V[:,1:r]);
    S = diagm(S[1:r]);

    Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
    h .= BCh(obj.settings,obj.SG,obj.sn,h,X*S*V',Temp);
    BT = aRad*AdvecSpeed.*Temp.^4 ./2/pi;

    Xnew = zeros(obj.T,size(X));
    Vnew = zeros(obj.T,size(V));
    Xtmp = zeros(obj.T,size(X));
    Vtmp = zeros(obj.T,size(V));

    # Pre- allocating memory to speed up computations
    VQxV = zeros(obj.T,r,r);
    VQyV = zeros(obj.T,r,r);
    VAbsQxV = zeros(obj.T,r,r);
    VAbsQyV = zeros(obj.T,r,r);

    XL2xX = zeros(obj.T,r,r);
    XL2yX = zeros(obj.T,r,r);
    XL1xX = zeros(obj.T,r,r);
    XL1yX = zeros(obj.T,r,r);

    unitvec = ones(obj.sn.TNv);
    AdvecMat = (I(obj.sn.TNv) - 1/2/pi .* obj.sn.w * Transpose(unitvec));
    
    II = zeros(Int,N_MicroGrid); J = zeros(Int,N_MicroGrid); vals = zeros(obj.T,N_MicroGrid);
    for k = 1:N_MicroGrid
        II[k] = k; J[k] = k; vals[k] = 1/(1/AdvecSpeed + dt*obj.SigmaT_mic[k,k]/ϵ^2);
    end
    Beta_K = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

    QxAM  = Qx*AdvecMat;
    QyAM = Qy*AdvecMat;
    AbsQxAM = AbsQx*AdvecMat;
    AbsQyAM = AbsQy*AdvecMat;
    Qx1 = unitvec'Qx;
    Qy1 = unitvec'Qy;

    BTh = zeros(obj.T,obj.SG.N_MacroGrid);
    BT_augx = zeros(obj.T,obj.SG.N_MicroGrid);
    BT_augy = zeros(obj.T,obj.SG.N_MicroGrid);
    Grad_g = zeros(obj.T,obj.SG.N_MacroGrid);

    ranks = zeros(Int64,Nt+1);
    ranks[1] = r;

    energy = zeros(Nt+1);
    mass = zeros(Nt+1);
    energy[1] = ComputeEnergy(obj,BT,Temp,h,X,S,V);
    mass[1] = ComputeMass(obj,BT,Temp,h);

    # save_solution(obj.settings.problem,"parallel_solver",Temp,0,0);
    material_idx = findall(x->x==100,obj.SigmaA_mac);

    for k = ProgressBar(1:Nt)
        
        BT_augx .= obj.SigmaT_mic_inv*deltaox*BT;
        BT_augy .= obj.SigmaT_mic_inv*deltaoy*BT;
        ## Micro update
        BTh .= BT .+ ϵ^2 .* h;

        # Augmenting the basis functions
        if obj.settings.problem == "Hohlraum"  

            Xaug = X;
            Vaug1,_ = py"qr"([w V]);
            Vaug = Vaug1[:,2:end];

            Saug = S * (V' *Vaug); #

            tot = r;
            totS = 2*tot;
        else
            Xaug,_ = py"qr"([BT_augx BT_augy X]);
            Vaug1,_ = py"qr"([w Qx*unitvec Qy*unitvec V]); # w 
            Vaug = Vaug1[:,2:end];

            Saug = (Xaug' * X) * S * (V' *Vaug);
            
            tot = r+2;
            totS = 2*tot;
            
        end        
        Beta = 1/AdvecSpeed .*I(tot) .+ dt/ϵ^2 .* Xaug'*obj.SigmaT_mic*Xaug;
        X1Tilde = zeros(size(Xaug))
        V1Tilde = zeros(size(Vaug))
        K = zeros(size(Xaug))
        L = zeros(size(Vaug))'

        @sync begin
            #K-step
            @async begin
                K .= Xaug*Saug;
                
                VQxV = Vaug'*QxAM*Vaug;
                VQyV = Vaug'*QyAM*Vaug;
                VAbsQxV = Vaug'*AbsQxAM*Vaug;
                VAbsQyV = Vaug'*AbsQyAM*Vaug;

                K .= K./AdvecSpeed .- dt/ϵ .*L2x * K * VQxV .+ dt/ϵ .*L1x * K * VAbsQxV  .- dt/ϵ .*L2y * K * VQyV  .+ dt/ϵ .*L1y * K * VAbsQyV .- dt/ϵ^2 .*deltaox*BTh*Qx1*Vaug .- dt/ϵ^2 .*deltaoy*BTh*Qy1*Vaug;
                K .= Beta_K*K;
                
                Xtmp,_ = py"qr"([Xaug K]);

                X1Tilde .= Xtmp[:,(tot+1):end];
            end

            @async begin
                #L-step
                L .= Saug*Vaug';

                XL2xX = Xaug'*L2x*Xaug;
                XL2yX = Xaug'*L2y*Xaug;
                XL1xX = Xaug'*L1x*Xaug;
                XL1yX = Xaug'*L1y*Xaug;

                L .= L./AdvecSpeed .- dt/ϵ .*XL2xX*L*QxAM .+ dt/ϵ .* XL1xX*L*AbsQxAM .- dt/ϵ .*XL2yX*L*QyAM .+ dt/ϵ .* XL1yX*L*AbsQyAM .- dt/ϵ^2 .*Xaug'*deltaox*BTh*Qx1 .- dt/ϵ^2 .*Xaug'*deltaoy*BTh*Qy1;
                
                L .= Beta\L;
                Vtmp,_ = py"qr"([w Vaug L']);
                V1Tilde .= Vtmp[:,(tot+2):end];
            end
    
            #S-step
            @async begin
                Saug .= Saug./AdvecSpeed .- dt/ϵ .*XL2xX*Saug*VQxV .+ dt/ϵ .* XL1xX*Saug*VAbsQxV .- dt/ϵ .*XL2yX*Saug*VQyV .+ dt/ϵ .* XL1yX*Saug*VAbsQyV .- dt/ϵ^2 .*Xaug'*deltaox*BTh*Qx1*Vaug .- dt/ϵ^2 .*Xaug'*deltaoy*BTh*Qy1*Vaug;
            end
        end

        S = zeros(totS,totS);
        S[1:(tot),1:(tot)] .= Beta\Saug;
        S[(tot+1):end,1:(tot)] .= X1Tilde' * K;
        S[1:(tot),(tot+1):end] .= L * V1Tilde;

        X = [Xaug X1Tilde];
        V = [Vaug V1Tilde];
        

        #Conservative truncation
        Khat = X * S;
        m = 2; # Number of basis vectors to left unchanged 
        Khat_ap, Khat_rem = Khat[:,1:m], Khat[:,m+1:end]; # Splitting Khat into basis required for Ap and remaining vectors
        Vap, Vrem = V[:,1:m], V[:,m+1:end]; # Splitting Khat into basis required for Ap and remaining vectors

        Xhrem, Shrem = qr(Khat_rem);
        Xhrem = Matrix(Xhrem);

        U, sigma, W = svd(Shrem);
        U = Matrix(U);
        W = Matrix(W);

        rmax = -1
        tmp = 0.0;
        tol = obj.settings.epsAdapt * norm(sigma);

        # Truncating the rank
        for i = 1:totS-m
            tmp = sqrt(sum(sigma[i:totS-m].^2));
            if tmp < tol
                rmax = i
                break
            end
        end
        rmaxTotal = Int(round(max(N_MicroGrid, obj.sn.TNv)/2));

        rmax = min(rmax,rmaxTotal);
        rmax = max(rmax,1);

        if rmax == -1
            rmax = rmaxTotal;
        end

        Uhat = U[:,1:rmax]; 
        What = W[:,1:rmax];
        sigma_hat = diagm(sigma[1:rmax]);

        W1 = Vrem * What;
        Xrem = Xhrem * Uhat;

        V = [Vap W1];
        Xap, Sap = py"qr"(Khat_ap);
        X, R2 = py"qr"([Xap Xrem]);

        S = zeros(rmax+m,rmax+m);
        S[1:m,1:m] = Sap;
        S[m+1:end,m+1:end] = sigma_hat;
        S .= R2 * S;
        r = rmax+m;
        ranks[k+1] = r;

        # ## Including Boundary condition into the problem
        X,S,V = BCg_lr(obj.settings,obj.SG,obj.sn,X,S,V,h,Temp);


        ## Macro-meso update
        Grad_g .= ((Dox * X * S )* (V' * Qx * w)) .+ ((Doy * X * S) * (V' * Qy * w));
        
        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY
                σᵢ = obj.SigmaA_mac[i,i];
                αᵢ = 1 + dt*AdvecSpeed*σᵢ/ϵ^2;
                rhsᵢ = density[i,i]*cᵥ*Temp[i] + 2*pi*dt*σᵢ/αᵢ*(h[i] + 1/ϵ^2*BT[i] - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
                f(x) = density[i,i]*cᵥ*x + (dt*aRad*AdvecSpeed*σᵢ/αᵢ/ϵ^2)*x^4 - rhsᵢ;
                Temp[i] = find_zero(f,Temp[i]);
            end
        end

        for i = 1:N_MacroGrid
            if i ∉ obj.SG.ghostidx_XY 
                αᵢ = 1 + dt*AdvecSpeed*obj.SigmaA_mac[i,i]/ϵ^2;
                BT1 = aRad*AdvecSpeed/2/pi*(Temp[i])^4;
                h[i] = 1/αᵢ*(h[i] - 1/ϵ^2*(BT1 - BT[i]) - dt*AdvecSpeed/2/pi/ϵ^2*Grad_g[i]);
            end
        end
        # BC
        Temp .= BCTemp(obj.SG,obj.settings.problem,Temp,Nx,Ny);
        h .= BCh(obj.settings,obj.SG,obj.sn,h,X*S*V',Temp);
        BT .= aRad*AdvecSpeed.*Temp.^4 ./2/pi;

        energy[k+1] = ComputeEnergy(obj,BT,Temp,h,X,S,V);
        mass[k+1] = ComputeMass(obj,BT,Temp,h);

        
        t = t + dt;
        # if k%10 == 0
        #     save_solution(obj.settings.problem,"parallel_solver",Temp,k,t);
        # end
    end
    sol = BundleSolution(obj,h,X*S*V',Temp,ranks, energy, mass);
    return sol;
end

