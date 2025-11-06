__precompile__

include("quadratures/standardQuadrature.jl")
include("quadratures/Quadrature.jl")

mutable struct SNSystem{T<:AbstractFloat}
    ## Diagonal Quadrature matrices
    Qx::SparseMatrixCSC{T, Int64};
    Qy::SparseMatrixCSC{T, Int64};
    Qz::SparseMatrixCSC{T, Int64};

    ## Roe stabilization matrices for the sptial discretisation
    AbsQx::SparseMatrixCSC{T, Int64};
    AbsQy::SparseMatrixCSC{T, Int64};
    AbsQz::SparseMatrixCSC{T, Int64};

    points::Array{T,2};
    w::Array{T};
    M::SparseMatrixCSC{T,Int64}; # Diagonal matrix of weights

    # Solver settings
    settings::Settings;

    # Quadrature ordes
    Nv::Int

    TNv::Int # Total number of quadrature points

    # Quadrature methods
    quadrature::Quadrature;

    T::DataType;

    # Constructor
    function SNSystem(settings, T::DataType=Float64)
        Nv = settings.Nv;
        TNv = Nv*Nv;

        quadrature = Quadrature(Nv,1);

        points = Array{T}(undef, 0, 0);
        w = [];
        M = sparse([],[],[],TNv,TNv);

        Qx = sparse([],[],[],TNv,TNv);
        Qy = sparse([],[],[],TNv,TNv);
        Qz = sparse([],[],[],TNv,TNv);

        AbsQx = sparse([],[],[],TNv,TNv);
        AbsQy = sparse([],[],[],TNv,TNv);
        AbsQz = sparse([],[],[],TNv,TNv);


        new{T}(Qx,Qy,Qz,AbsQx,AbsQy,AbsQz,points,w,M,settings,Nv,TNv,quadrature,T);
    end
end

function SetupSNprojected2D(obj::SNSystem)
    Nv = obj.settings.Nv;
    TNv = obj.TNv; # Total number of quadrature points

    points,weights = computeXYZandWeightsProjected2D(Nv);

    IIx = zeros(TNv); Jx = zeros(TNv); valsx =zeros(TNv); valsabsx =zeros(TNv);
    IIy = zeros(TNv); Jy = zeros(TNv); valsy =zeros(TNv); valsabsy =zeros(TNv);
    IIz = zeros(TNv); Jz = zeros(TNv); valsz =zeros(TNv); valsabsz =zeros(TNv);
    IIM = zeros(TNv); JM = zeros(TNv); valsM =zeros(TNv); 

    for k = 1:TNv
        IIx[k] = k; IIy[k] = k; IIz[k] = k; IIM[k] = k;
        Jx[k] = k; Jy[k] = k; Jz[k] = k; JM[k] = k;
        valsx[k] = points[k,1]; valsy[k] = points[k,2]; valsz[k] = points[k,3]; valsM[k] = weights[k];
        valsabsx[k] = abs(points[k,1]); valsabsy[k] = abs(points[k,2]); valsabsz[k] = abs(points[k,3]);
    end
    obj.Qx = sparse(IIx,Jx,valsx,TNv,TNv);
    obj.Qy = sparse(IIy,Jy,valsy,TNv,TNv);
    obj.Qz = sparse(IIz,Jz,valsz,TNv,TNv);

    obj.AbsQx = sparse(IIx,Jx,valsabsx,TNv,TNv);
    obj.AbsQy = sparse(IIy,Jy,valsabsy,TNv,TNv);
    obj.AbsQz = sparse(IIz,Jz,valsabsz,TNv,TNv);

    obj.points = points;
    obj.w = weights;
    obj.M = sparse(IIM,JM,valsM,TNv,TNv);
end