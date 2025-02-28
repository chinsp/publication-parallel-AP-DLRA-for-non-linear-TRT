mutable struct StaggeredGrid{T<:AbstractFloat}
    settings::Settings;

    # Number of grid points in the staggered grid
    N_MicroGrid::Int64;
    N_MacroGrid::Int64;

    # Cartesian product of the grids
    XMidY::Array{Tuple{T,T},2};
    XYMid::Array{Tuple{T,T},2};
    XY::Array{Tuple{T,T},2};
    XMidYMid::Array{Tuple{T,T},2};

    # Micro and Macro grid points flattened according to setup
    micro_grid::Array{Tuple{T,T},1};
    macro_grid::Array{Tuple{T,T},1};

    ## Indices of the of micro/macro grid corresponding to the Cartesian products

    idx_XMidY::Array{Int64,1};
    idx_XYMid::Array{Int64,1};
    idx_XY::Array{Int64,1};
    idx_XMidYMid::Array{Int64,1};

    # Ghost cell indices
    ghostidx_XMidY;
    ghostidx_XYMid;
    ghostidx_XY;

    # Specific ghost cells microgrid
    ghostidx_mic_L;
    ghostidx_mic_R;
    ghostidx_mic_T;
    ghostidx_mic_B;

    # Specific ghost cells macrogrid
    ghostidx_mac_L;
    ghostidx_mac_R;
    ghostidx_mac_T;
    ghostidx_mac_B;

    ## Boundary cell indices
    boundaryidx_micro;
    boundaryidx_macro;

    # Specific boundaries microgrid
    boundaryidx_mic_L; # Left
    boundaryidx_mic_R; # Right
    boundaryidx_mic_T; # Top
    boundaryidx_mic_B; # Bottom

    # Specific boundaries macrogrid
    boundaryidx_mac_L; # Left
    boundaryidx_mac_R; # Right
    boundaryidx_mac_T; # Top
    boundaryidx_mac_B; # Bottom

    function StaggeredGrid(settings,T::DataType=Float64)
        N_MicroGrid = N_interfaces(settings);
        N_MacroGrid = N_corners_midPoints(settings);

        XMidY = CartesianProduct(settings.xMid,settings.y);
        XYMid = CartesianProduct(settings.x,settings.yMid);
        XY = CartesianProduct(settings.x,settings.y);
        XMidYMid = CartesianProduct(settings.xMid,settings.yMid);

        micro_grid,idx_XMidY,idx_XYMid,ghostidx_XMidY,ghostidx_XYMid,boundaryidx_micro,boundaryidx_mic_L,boundaryidx_mic_R,boundaryidx_mic_T,boundaryidx_mic_B,ghostidx_mic_L,ghostidx_mic_R,ghostidx_mic_T,ghostidx_mic_B = Setup_MicroGrid(settings);
        macro_grid,idx_XMidYMid,idx_XY,ghostidx_XY,boundaryidx_macro,boundaryidx_mac_L,boundaryidx_mac_R,boundaryidx_mac_T,boundaryidx_mac_B,ghostidx_mac_L,ghostidx_mac_R,ghostidx_mac_T,ghostidx_mac_B = Setup_MacroGrid(settings);
        
        new{T}(settings,N_MicroGrid,N_MacroGrid,XMidY,XYMid,XY,XMidYMid,micro_grid,macro_grid,idx_XMidY,idx_XYMid,idx_XY,idx_XMidYMid,ghostidx_XMidY,ghostidx_XYMid,ghostidx_XY,
        ghostidx_mic_L,ghostidx_mic_R,ghostidx_mic_T,ghostidx_mic_B,ghostidx_mac_L,ghostidx_mac_R,ghostidx_mac_T,ghostidx_mac_B,boundaryidx_micro,boundaryidx_macro,boundaryidx_mic_L,boundaryidx_mic_R,boundaryidx_mic_T,boundaryidx_mic_B,boundaryidx_mac_L,boundaryidx_mac_R,boundaryidx_mac_T,boundaryidx_mac_B);
    end

end

function N_interfaces(obj::Settings)
    Nx = obj.Nx;
    Ny = obj.Ny;
    return Int((Nx-1)*Ny + Nx*(Ny-1));
end

function N_corners_midPoints(obj::Settings)
    Nx = obj.Nx;
    Ny = obj.Ny;
    return Int((Nx-1)*Ny + Nx*(Ny-1) - 3) ;
end

function CartesianProduct(x::Array,y::Array)
    prod = [(a,b) for a in x, b in y];
    return prod;
end

function Setup_MicroGrid(obj::Settings)
    x = obj.x;
    xMid = obj.xMid;
    y = obj.y;
    yMid = obj.yMid;

    dim_interfaces = N_interfaces(obj);
    XMidY = CartesianProduct(xMid,y);
    XYMid = CartesianProduct(x,yMid);
    micro_grid = Vector(undef,dim_interfaces);

    index_XMidY = zeros(Int,(obj.Nx-1)*obj.Ny);
    index_XYMid = zeros(Int,obj.Nx*(obj.Ny-1));

    ghostidx_XMidY = [];
    ghostidx_XYMid =  [];

    boundaryidx_micro = [];
    boundaryidx_mic_L = [];
    boundaryidx_mic_R = [];
    boundaryidx_mic_T = [];
    boundaryidx_mic_B = [];
    
    ghostidx_mic_L = [];
    ghostidx_mic_R = [];
    ghostidx_mic_T = [];
    ghostidx_mic_B = [];

    counter = 0
    for j = 1:obj.Ny
        micro_grid[counter+1:counter+obj.Nx-1] = XMidY[:,j];
        index_XMidY[(j-1)*(obj.Nx-1)+1:j*(obj.Nx-1)] = counter+1:counter+obj.Nx-1
        if j == 1 || j == obj.Ny
            append!(ghostidx_XMidY,collect(counter+1:counter+obj.Nx-1));
        end
        counter += obj.Nx+obj.Nx-1;
    end
    counter = obj.Nx-1
    for j = 1:obj.Ny-1
        micro_grid[counter+1:counter+obj.Nx] = XYMid[:,j];
        index_XYMid[(j-1)*(obj.Nx)+1:j*(obj.Nx)] = counter+1:counter+obj.Nx;
        append!(ghostidx_XYMid,counter+1);
        append!(ghostidx_XYMid,counter+obj.Nx);
        counter += obj.Nx+obj.Nx-1;
    end
    for k = eachindex(micro_grid)
        if (micro_grid[k][1] == xMid[1] || micro_grid[k][1] == xMid[end]) && micro_grid[k][2] ∈ y[2:end-1]
            append!(boundaryidx_micro,k);
        elseif (micro_grid[k][2] == yMid[1] || micro_grid[k][2] == yMid[end]) && micro_grid[k][1] ∈ x[2:end-1]
            append!(boundaryidx_micro,k);
        end
    end
    for k = eachindex(micro_grid)
        if micro_grid[k][1] == xMid[1] && micro_grid[k][2] ∈ y[2:end-1]
            append!(boundaryidx_mic_L,k);
        elseif micro_grid[k][1]== xMid[end] && micro_grid[k][2] ∈ y[2:end-1]
            append!(boundaryidx_mic_R,k);
        end
        if micro_grid[k][2]== yMid[end] && micro_grid[k][1] ∈ x[2:end-1]
            append!(boundaryidx_mic_T,k);
        elseif micro_grid[k][2]== yMid[1] && micro_grid[k][1] ∈ x[2:end-1]
            append!(boundaryidx_mic_B,k);
        end
    end
    for k = eachindex(micro_grid)
        if micro_grid[k][1] == x[1] && micro_grid[k][2] ∈ yMid
            append!(ghostidx_mic_L,k);
        elseif micro_grid[k][1]== x[end] && micro_grid[k][2] ∈ yMid
            append!(ghostidx_mic_R,k);
        end
        if micro_grid[k][2]== y[end] && micro_grid[k][1] ∈ xMid
            append!(ghostidx_mic_T,k);
        elseif micro_grid[k][2]== y[1] && micro_grid[k][1] ∈ xMid
            append!(ghostidx_mic_B,k);
        end
    end

    return micro_grid,index_XMidY,index_XYMid,ghostidx_XMidY,ghostidx_XYMid,boundaryidx_micro,boundaryidx_mic_L,boundaryidx_mic_R,boundaryidx_mic_T,boundaryidx_mic_B,ghostidx_mic_L,ghostidx_mic_R,ghostidx_mic_T,ghostidx_mic_B;
end

function Setup_MacroGrid(obj::Settings)
    x = obj.x;
    xMid = obj.xMid;
    y = obj.y;
    yMid = obj.yMid;

    dim_macrogrid = N_corners_midPoints(s);
    XY = CartesianProduct(x,y);
    XMidYMid = CartesianProduct(xMid,yMid);
    macro_grid = Vector(undef,dim_macrogrid);

    index_XMidYMid = zeros(Int,(obj.Nx-1)*(obj.Ny-1));

    ghost_L = CartesianProduct([x[1]],y[2:end-1]);
    ghost_R = CartesianProduct([x[end]],y[2:end-1]);
    ghost_T = CartesianProduct(x[2:end-1],[y[end]]);
    ghost_B = CartesianProduct(x[2:end-1],[y[1]]);
    ghost_XY = union(ghost_L,ghost_R,ghost_T,ghost_B);
    
    boundary_L = CartesianProduct([xMid[1]],yMid);
    boundary_R = CartesianProduct([xMid[end]],yMid);
    boundary_T = CartesianProduct(xMid,[yMid[end]]);
    boundary_B = CartesianProduct(xMid,[yMid[1]]);
    boundary_macro = union(boundary_L,boundary_R,boundary_T,boundary_B);


    counter = 0
    for j = 1:obj.Ny
        if j == 1
            macro_grid[counter+1:counter+obj.Nx-2] = XY[2:end-1,j];
            counter += obj.Nx+obj.Nx-3;

        elseif j == obj.Ny
            macro_grid[end-obj.Nx+3:end] = XY[2:end-1,j];
        else
            macro_grid[counter+1:counter+obj.Nx] = XY[:,j];
            counter += obj.Nx+obj.Nx-1;
        end
        
    end

    counter = obj.Nx-2
    for j = 1:obj.Ny-1
        macro_grid[counter+1:counter+obj.Nx-1] = XMidYMid[:,j];
        index_XMidYMid[(j-1)*(obj.Nx-1)+1:j*(obj.Nx-1)] = counter+1:counter+obj.Nx-1;
        counter += obj.Nx+obj.Nx-1;
    end

    ghostidx_XY = (findall(z->z ∈ ghost_XY,macro_grid));
    index_XY = (findall(z->z ∈ XY,macro_grid));
    boundaryidx_macro = (findall(z->z ∈ boundary_macro,macro_grid));
    boundaryidx_mac_L = (findall(z->z ∈ boundary_L,macro_grid));
    boundaryidx_mac_R = (findall(z->z ∈ boundary_R,macro_grid));
    boundaryidx_mac_T = (findall(z->z ∈ boundary_T,macro_grid));
    boundaryidx_mac_B = (findall(z->z ∈ boundary_B,macro_grid));
    ghostidx_mac_L = (findall(z->z ∈ ghost_L,macro_grid));
    ghostidx_mac_R = (findall(z->z ∈ ghost_R,macro_grid));
    ghostidx_mac_T = (findall(z->z ∈ ghost_T,macro_grid));
    ghostidx_mac_B = (findall(z->z ∈ ghost_B,macro_grid));

    return macro_grid,index_XMidYMid,index_XY,ghostidx_XY,boundaryidx_macro,boundaryidx_mac_L,boundaryidx_mac_R,boundaryidx_mac_T,boundaryidx_mac_B,ghostidx_mac_L,ghostidx_mac_R,ghostidx_mac_T,ghostidx_mac_B;
end