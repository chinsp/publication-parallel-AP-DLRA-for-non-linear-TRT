__precompile__

mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    Ny::Int64;
    # number spatial cells
    NCellsX::Int64;
    NCellsY::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    c::Float64;
    d::Float64;
    # grid cell width
    dx::Float64
    dy::Float64

    # Temporal settings
    tEnd::Float64; # End time
    dt::Float64; # Time step
    cfl::Float64; # CFL number 
    cfltype::String; # Hyperbolic or Parabolic CFL
    
    # Number of quadrature points
    Nv::Int64;

    # Spatial grid
    x
    xMid
    y
    yMid

    # Problem definitions
    problem::String;

    # Scaling parameters
    epsilon::Float64;

    # Physical parameters
    sigmaA::Float64;
    sigmaS::Float64;   
    sigmaT::Float64; 
    AdvecSpeed::Float64;
    c_nu::Float64;
    aRad::Float64;
    
    # rank
    r::Int;
    epsAdapt::Float64;

    # Boundary condition type
    BCType::Int64;

    function Settings(Nx::Int=102,Ny::Int=102,r::Int=15,epsilon::Float64=1.0,problem::String="Gaussian",cfltype::String="h")

        # Spatial grid setting
        NCellsX = Nx - 1;
        NCellsY = Ny - 1;
        
        if problem == "Gaussian" || problem == "MarshakWave"
            a = 0.0; # Left boundary
            b = 0.002; # Right boundary
    
            c = 0.0; # Lower boundary
            d = 0.002; # Upper boundary   

            # Physical parameters
            sigmaS = 0.0;
            sigmaA = 1.0 / 0.926 / 1e-6 / 100.0;
            sigmaT = sigmaA + sigmaS;
            AdvecSpeed = 299792458.0 * 100.0;                # speed of light in [cm/s]
            # aRad = 4 * StefBoltz / c;     # Radiation constant
            aRad = 7.565767381646406e-15;
            density = 0.01;
            c_nu = density * 0.831 * 1e7;    # heat capacity: [kJ/(kg K)] = [1000 m^2 / s^2 / K] therefore density * 0.831 * 1e7        

            # Quadrature order on the disk
            Nv = 30; # Should be even for projected 2D quadrature

            tEnd = 0.05 * 1e-10;
            # tEnd = 0.01 * 1e-10;
            BCType = 1;
        elseif problem == "Hohlraum"
            a = 0.0; # Left boundary
            b = 1.0; # Right boundary
    
            c = 0.0; # Lower boundary
            d = 1.0; # Upper boundary    

            # Lower bounds for the physical parameters, the matrices are set up in the solver
            sigmaS = 0.0;
            sigmaA = 0.0;# 0.5;
            sigmaT = sigmaA + sigmaS;
            AdvecSpeed = 29.9792458;                # speed of light in [cm/ns]
            # aRad = 4*StefBoltz/c;     # Radiation constant
            aRad =  0.01372;  # Radiation constant 
            density = 0.01;
            c_nu = 0.3;    # heat capacity

            # Quadrature order on the disk
            Nv = 30; # Should be even for projected 2D quadrature

            tEnd = 1.0;
            BCType = 2;
        end

        # Spatial grid
        x = collect(range(a,stop = b,length = NCellsX));
        dx = x[2]-x[1];
        x = [x[1]-dx;x]; # Add ghost cells so that boundary cell centers lie on a and b
        x = x.+dx/2;
        xMid = x[1:(end-1)].+0.5*dx

        y = collect(range(c,stop = d,length = NCellsY));
        dy = y[2]-y[1];
        y = [y[1]-dy;y]; # Add ghost cells so that boundary cell centers lie on a and b
        y = y.+dy/2;
        yMid = y[1:(end-1)].+0.5*dy

        # Time settings

        cfl = 1.0 # CFL condition
        dt = cfl*dx; # The step-size is set in the solver

        epsAdapt = 1e-2;

        # build class
        new(Nx,Ny,NCellsX,NCellsY,a,b,c,d,dx,dy,tEnd,dt,cfl,cfltype,Nv,x,xMid,y,yMid,problem,epsilon,sigmaA,sigmaS,sigmaT,AdvecSpeed,c_nu,aRad,r,epsAdapt,BCType);
    end
end

function ICTemp(obj::Settings,x,boundaryidx,ghostidx)
    out = zeros(length(x));
    if obj.problem == "Gaussian"
        for k = eachindex(out)
            x₀,y₀ = 0.001,0.001
            xᵢ,yᵢ = x[k][1],x[k][2];
            σₓ =  1e-4;
            out[k] = (1/2/pi/σₓ^2)*exp(-((xᵢ-x₀)^2 + (yᵢ-y₀)^2)/2/σₓ^2);
        end
        max_temp = maximum(out);
        for k = eachindex(out)
            if out[k] < 0.02
                out[k] = 0.02
            else
                out[k] = 80*out[k]/max_temp
            end
        end
        out = 11604.0 .* out;
    elseif obj.problem == "MarshakWave"
        for k = eachindex(out)
            if k ∉ ghostidx
                out[k] = 0.02 * 11604.0
            end
        end
    elseif obj.problem == "Hohlraum"
        for k = eachindex(out)
            if k ∉ ghostidx
                out[k] = 1e-3;
            end
        end
    else
        println("The initial condition has not been coded please enter valid code for the initial condition")
        println("Currently available:")
        println("1. Gaussian")
        println("2. MarshakWave")
        println("3. Hohlraum")
    end
    return out;
end

function ICParDen(obj::Settings,xmic,xmac,quadpoints,ghost_micro,Temp)
    out1 = zeros(length(xmic),size(quadpoints)[1]);
    out2 = zeros(length(xmac));
    # end
    return out1,out2;
end

function ICh(obj::Settings,scalflux,Temp)
    out = zeros(length(Temp));
    return out;
end

function ICg(obj::Settings,parden,scalflux)
    N_MicroGrid,TNv = size(parden);
    out = zeros(N_MicroGrid,TNv);
    return out;
end