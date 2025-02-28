using SparseArrays

# include("utils.jl")
# include("settings.jl")
# include("StaggeredGrid.jl")

struct Stencils{T<:AbstractFloat}
    L1x::SparseMatrixCSC{T, Int64};
    L1y::SparseMatrixCSC{T, Int64};
    L2x::SparseMatrixCSC{T, Int64};
    L2y::SparseMatrixCSC{T, Int64};

    deltaox::SparseMatrixCSC{T, Int64};
    deltaoy::SparseMatrixCSC{T, Int64};
    Dox::SparseMatrixCSC{T, Int64};
    Doy::SparseMatrixCSC{T, Int64};

    function Stencils(settings::Settings,T::DataType=Float64)
        # density = settings.density;
        # setup stencil matrix
        nx = settings.Nx;
        ny = settings.Ny;
        dx = settings.dx;
        dy = settings.dy;

        SG = StaggeredGrid(settings);
        N_MicroGrid = SG.N_MicroGrid;
        N_MacroGrid = SG.N_MacroGrid;

        ghostidx_XMidY = SG.ghostidx_XMidY;
        idx_XMidY = SG.idx_XMidY;
        idx_XYMid = SG.idx_XYMid;
        idx_XY = SG.idx_XY;
        idx_XMidYMid = SG.idx_XMidYMid;

        ghostidx_micro = union(SG.ghostidx_XMidY,SG.ghostidx_XYMid);
        NGidx_micro = setdiff(1:N_MicroGrid,ghostidx_micro);
        ghostidx_macro = SG.ghostidx_XY;
        NGidx_macro = setdiff(1:N_MacroGrid,ghostidx_macro);

        L1x = spzeros(N_MicroGrid,N_MicroGrid);
        L1y = spzeros(N_MicroGrid,N_MicroGrid);
        L2x = spzeros(N_MicroGrid,N_MicroGrid);
        L2y = spzeros(N_MicroGrid,N_MicroGrid);

        deltaox = spzeros(N_MicroGrid,N_MacroGrid);
        deltaoy = spzeros(N_MicroGrid,N_MacroGrid);
        Dox = spzeros(N_MacroGrid,N_MicroGrid);
        Doy = spzeros(N_MacroGrid,N_MicroGrid);

        II = zeros(2*(nx-1)*(ny-2) + 2*(nx-2)*(ny-1)); J = zeros(2*(nx-1)*(ny-2) + 2*(nx-2)*(ny-1)); vals =zeros(2*(nx-1)*(ny-2) + 2*(nx-2)*(ny-1));
        l = 1;
        for k in NGidx_micro
            idx = k;
            idxplus = idx+1;
            idxminus = idx-1;
            if idx ∈ SG.boundaryidx_mic_L
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dx; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -1.0/2/dx; l += 1; # Using one-sided differences near the boundaries
            elseif idx ∈ SG.boundaryidx_mic_R
                II[l] = idx; J[l] = idxminus; vals[l] = -1.0/2/dx; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = 1.0/2/dx; l += 1; # Using one-sided differences near the boundaries
            else
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dx; l += 1;
                II[l] = idx; J[l] = idxminus; vals[l] = -1.0/2/dx; l += 1; # Using central differences in the interior of domain
            end
        end
        L2x = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

        II = zeros(2*(nx-1)*(ny-2) + 2*(nx-2)*(ny-1)); J = zeros(2*(nx-1)*(ny-2) + 2*(nx-2)*(ny-1)); vals =zeros(2*(nx-1)*(ny-2) + 2*(nx-2)*(ny-1));
        l = 1;
        for k in NGidx_micro
            idx = k;
            idxplus = min(N_MicroGrid,idx+(nx+nx-1));
            idxminus = max(1,idx-(nx+nx-1));
            if idx ∈ SG.boundaryidx_mic_B
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dy; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -1.0/2/dy; l += 1; # Using one-sided differences near the boundaries
-            elseif idx ∈ SG.boundaryidx_mic_T
                II[l] = idx; J[l] = idxminus; vals[l] = -1.0/2/dy; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = 1.0/2/dy; l += 1; # Using one-sided differences near the boundaries
            else
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dy; l += 1;
                II[l] = idx; J[l] = idxminus; vals[l] = -1.0/2/dy; l += 1; # Using central differences in the interior of domain
            end
        end
        L2y = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

        II = zeros(4*(ny-2) + 3*(nx-3)*(ny-2) + 3*(nx-2)*(ny-1)); J = zeros(4*(ny-2) + 3*(nx-3)*(ny-2) + 3*(nx-2)*(ny-1)); vals =zeros(4*(ny-2) + 3*(nx-3)*(ny-2) + 3*(nx-2)*(ny-1));
        l =1;
        for k in NGidx_micro
            idx = k;
            idxplus = idx+1;
            idxminus = idx-1;
            if idx ∈ SG.boundaryidx_mic_L
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dx; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -1.0/2/dx; l += 1;
            elseif idx ∈ SG.boundaryidx_mic_R
                II[l] = idx; J[l] = idxminus; vals[l] = 1.0/2/dx; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -1.0/2/dx; l += 1;
            else
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dx; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -2.0/2/dx; l += 1;
                II[l] = idx; J[l] = idxminus; vals[l] = 1.0/2/dx; l += 1;
            end
        end
        L1x = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

        II = zeros(4*(nx-2) + 3*(ny-3)*(nx-2) + 3*(ny-2)*(nx-1)); J = zeros(4*(nx-2) + 3*(ny-3)*(nx-2) + 3*(ny-2)*(nx-1)); vals =zeros(4*(nx-2) + 3*(ny-3)*(nx-2) + 3*(ny-2)*(nx-1));
        l = 1;
        for k in NGidx_micro
            idx = k;
            idxplus = min(N_MicroGrid,idx+(nx+nx-1));
            idxminus = max(1,idx-(nx+nx-1));
            if idx ∈ SG.boundaryidx_mic_B
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dy; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -1.0/2/dy; l += 1;
            elseif idx ∈ SG.boundaryidx_mic_T
                II[l] = idx; J[l] = idxminus; vals[l] = 1.0/2/dy; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -1.0/2/dy; l += 1;
            else
                II[l] = idx; J[l] = idxplus; vals[l] = 1.0/2/dy; l += 1;
                II[l] = idx; J[l] = idx; vals[l] = -2.0/2/dy; l += 1;
                II[l] = idx; J[l] = idxminus; vals[l] = 1.0/2/dy; l += 1;
            end
        end
        L1y = sparse(II,J,vals,N_MicroGrid,N_MicroGrid);

        ## Defining matrices between micro and macro grid
        II = zeros(2*size(NGidx_macro)[1]); J = zeros(2*size(NGidx_macro)[1]); vals =zeros(2*size(NGidx_macro)[1]);
        l = 1;
        for k in NGidx_macro
            idx = k;
            idxplus = idx + 2; 
            idxminus = idx + 1;
            II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dx; l += 1;
            II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dx; l += 1;
        end
        Dox = sparse(II,J,vals,N_MacroGrid,N_MicroGrid);
        
        II = zeros(2*size(NGidx_macro)[1]); J = zeros(2*size(NGidx_macro)[1]); vals =zeros(2*size(NGidx_macro)[1]);
        l = 1;
        for k in NGidx_macro
            idx = k;
            idxplus = idx + (nx+1); 
            idxminus = idx - (nx-2);
            II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dy; l += 1;
            II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dy; l += 1;
        end
        Doy = sparse(II,J,vals,N_MacroGrid,N_MicroGrid);
        
        II = zeros(2*(ny-1)*(nx-2) + 2*(nx-1)*(ny-2) ); J = zeros(2*(ny-1)*(nx-2) + 2*(nx-1)*(ny-2) ); vals =zeros(2*(ny-1)*(nx-2) + 2*(nx-1)*(ny-2)); #- 2*(ny-2)
        l = 1;
        for k in NGidx_micro
            idx = k;
            idxplus = idx -1; 
            idxminus = idx - 2;
            # if idxplus in ghostidx_macro
            #     II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dy; l += 1;    
            # elseif idxminus in ghostidx_macro
            #     II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dy; l += 1;
            # else
            #     II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dy; l += 1;
            #     II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dy; l += 1;
            # end
            II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dx; l += 1;
            II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dx; l += 1;
        end
        deltaox = sparse(II,J,vals,N_MicroGrid,N_MacroGrid);

        II = zeros(2*(nx-1)*(ny-2) + 2*(ny-1)*(nx-2)); J = zeros(2*(nx-1)*(ny-2) + 2*(ny-1)*(nx-2) ); vals =zeros(2*(nx-1)*(ny-2) + 2*(ny-1)*(nx-2) ); #- 2*(nx-2)
        l = 1;
        for k in NGidx_micro
            idx = k;
            if idx in SG.boundaryidx_mic_B
                idxplus = idx + (nx - 2); 
                idxminus = idx - nx;
            elseif idx in SG.boundaryidx_mic_T
                idxplus = idx + (nx-3); 
                idxminus = idx - (nx+1);
            else
                idxplus = idx + (nx-2); 
                idxminus = idx - (nx+1);
            end
            # if idxplus in ghostidx_macro
            #     II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dy; l += 1;    
            # elseif idxminus in ghostidx_macro
            #     II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dy; l += 1;
            # else
            #     II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dy; l += 1;
            #     II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dy; l += 1;
            # end
            II[l] = idx; J[l] = idxplus; vals[l] = 1.0/dy; l += 1;
            II[l] = idx; J[l] = idxminus; vals[l] = -1.0/dy; l += 1;
        end
        deltaoy = sparse(II,J,vals,N_MicroGrid,N_MacroGrid);

        new{T}(L1x,L1y,L2x,L2y,deltaox,deltaoy,Dox,Doy)
    end
end