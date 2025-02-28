## This is a julia code to implement boundary conditions for the solvers based on the 
## specific problems


function BCTemp(obj::StaggeredGrid,problem,Temp,Nx,Ny)
    if problem == "Gaussian"
        for k in obj.boundaryidx_macro
            Temp[k] = 0.02 * 11604.0;
        end
        for k in obj.ghostidx_XY
            Temp[k] = 0.02 * 11604.0;
        end
    elseif problem == "MarshakWave"
        for k in obj.boundaryidx_macro
            if k ∈ obj.boundaryidx_mac_L
                Temp[k] = 80 * 11604.0;
            end
            if k ∈ obj.boundaryidx_mac_R
                Temp[k] = 0.02 * 11604.0;
            end
            if k ∈ obj.boundaryidx_mac_T[2:end]
                Temp[k] = 0.02 * 11604.0;
            end
            if k ∈ obj.boundaryidx_mac_B[2:end]
                Temp[k] = 0.02 * 11604.0;
            end
        end

        for k in obj.ghostidx_XY
            if k ∈ obj.ghostidx_mac_L
                Temp[k] = 80 * 11604.0;
            end
            if k ∈ obj.ghostidx_mac_R
                Temp[k] = 0.02 * 11604.0;
            end
            if k ∈ obj.ghostidx_mac_T
                Temp[k] = 0.02 * 11604.0;
            end
            if k ∈ obj.ghostidx_mac_B
                Temp[k] = 0.02 * 11604.0;
            end
        end
    elseif problem == "Hohlraum"
        for k ∈ obj.boundaryidx_mac_L
            Temp[k] = 1; 
        end
        for k ∈ obj.boundaryidx_mac_R
            Temp[k] = 1e-3;
        end 
        for k in obj.ghostidx_XY
            Temp[k] = 0.0;
        end
    end

    return Temp;
end

function BCh(obj::Settings,obj1::StaggeredGrid, obj2::SNSystem,h,g,T)
    Intg_outflow_R = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,1] > 0.0
            Intg_outflow_R[q,q] = 1.0;
        end
    end
    Intg_outflow_T = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,2] > 0.0
            Intg_outflow_T[q,q] = 1.0;
        end
    end
    Intg_outflow_L = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,1] < 0.0
            Intg_outflow_L[q,q] = 1.0;
        end
    end
    Intg_outflow_B = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,2] < 0.0
            Intg_outflow_B[q,q] = 1.0;
        end
    end
    # h1 = h;

    if obj.BCType == 0    
        h[obj1.boundaryidx_macro] .= 0.0;
        h[obj1.ghostidx_XY] .= 0.0;
    elseif obj.BCType == 1
        # Temperature has been described on all the boundaries and the inflow particle density is in equilibrium 
        # with the temperature, i.e. f(t,x,v) = B(T(x)) n.v < 0, where B is given by Stefan-Boltzmann law

        for i ∈ obj1.boundaryidx_macro
            if i ∈ obj1.boundaryidx_mac_R[2:end-1]
                h[i] = 1/pi/obj.epsilon * obj2.w' * Intg_outflow_R * g[i+1,:];
            elseif i ∈ obj1.boundaryidx_mac_L
                h[i] = 1/pi/obj.epsilon * obj2.w' * Intg_outflow_L * g[i+2,:];
            elseif i ∈ obj1.boundaryidx_mac_B[2:end]
                h[i] = 1/pi/obj.epsilon * obj2.w' * Intg_outflow_B * g[i+(obj.Nx+1),:];
            elseif i ∈ obj1.boundaryidx_mac_T[2:end]
                h[i] = 1/pi/obj.epsilon * obj2.w' * Intg_outflow_T * g[i-(obj.Nx-2),:];
            end
        end

        for i ∈ obj1.ghostidx_XY
            if i ∈ obj1.ghostidx_mac_R
                h[i] = -h[i-1] + 2/pi/obj.epsilon * obj2.w' * Intg_outflow_R * (g[i+1,:]);
            elseif i ∈ obj1.ghostidx_mac_L
                h[i] = -h[i+1] + 2/pi/obj.epsilon * obj2.w' * Intg_outflow_L * (g[i+2,:]);
            elseif i ∈ obj1.ghostidx_mac_B
                h[i] = -h[i+(2*obj.Nx-2)] + 2/pi/obj.epsilon * obj2.w' * Intg_outflow_L * (g[i+(obj.Nx),:]);
            elseif i ∈ obj1.ghostidx_mac_T
                h[i] = -h[i-((2*obj.Nx-2))] + 2/pi/obj.epsilon * obj2.w' * Intg_outflow_L * (g[i-(obj.Nx-3),:]);
            end
        end

    elseif obj.BCType == 2
        # Temperature has not been described at left and right boundaries and the inflow particle density is in equilibrium 
        # with the temperature, i.e. f(t,x,v) = B(T(x)) n.v < 0, where B is given by Stefan-Boltzmann law. The top and bottom
        # boundaries are reflective.
        for i ∈ obj1.boundaryidx_macro
            if i ∈ obj1.boundaryidx_mac_R[2:end-1]
                h[i] = 1/pi/obj.epsilon * obj2.w' * Intg_outflow_R * (g[i+1,:]);
            elseif i ∈ obj1.boundaryidx_mac_L[2:end-1]
                h[i] = 1/pi/obj.epsilon * obj2.w' * Intg_outflow_L * (g[i+2,:]);
            end
        end

        for i ∈ obj1.ghostidx_XY
            if i ∈ obj1.ghostidx_mac_R[2:end-1]
                h[i] = -h[i-1] + 2/pi/obj.epsilon * obj2.w' * Intg_outflow_R * (g[i+1,:]);
            elseif i ∈ obj1.ghostidx_mac_L[2:end-1]
                h[i] = -h[i+1] + 2/pi/obj.epsilon * obj2.w' * Intg_outflow_L * (g[i+2,:]);
            end
        end
    else
        println("Not coded yet")
    end
    return h;
end

function BCg(obj::Settings,obj1::StaggeredGrid, obj2::SNSystem,g,h,T)
    TNv = obj2.TNv;
    g_wBC = zeros(size(g));
    g_wBC .= g;
    Intg_outflow_R = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,1] > 0.0
            Intg_outflow_R[q,q] = 1.0;
        end
    end
    Intg_outflow_T = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,2] > 0.0
            Intg_outflow_T[q,q] = 1.0;
        end
    end
    Intg_outflow_L = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,1] < 0.0
            Intg_outflow_L[q,q] = 1.0;
        end
    end
    Intg_outflow_B = zeros(obj2.TNv,obj2.TNv);
    for q = 1:obj2.TNv
        if obj2.points[q,2] < 0.0
            Intg_outflow_B[q,q] = 1.0;
        end
    end
    ghostidx_micro = union(obj1.ghostidx_XMidY,obj1.ghostidx_XYMid)
    
    w = obj2.w;

    if obj.BCType == 0    
        g_wBC[obj1.boundaryidx_micro,:] .= 0.0;
        g_wBC[ghostidx_micro,:] .= 0.0;
    elseif obj.BCType == 1
        # Temperature has been described on all the boundaries and the inflow particle density is in equilibrium 
        # with the temperature, i.e. f(t,x,v) = B(T(x)) n.v < 0, where B is given by Stefan-Boltzmann law. These 
        # boundary conditions are derived by using consistency conditions for the scheme
        for i in obj1.boundaryidx_micro
            if i ∈ obj1.boundaryidx_mic_L
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        g_wBC[i,j] = -obj.epsilon*(h[i-1]+h[i-2])/2 # - (obj.aRad*obj.AdvecSpeed*T[i-1]^4/2/pi)/2/obj.epsilon;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_R
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        g_wBC[i,j] = -obj.epsilon*(h[i-2]+h[i-1])/2 # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_T
                for j = 1:obj2.TNv
                    if obj2.points[j,2] < 0.0
                        g_wBC[i,j] = -obj.epsilon*(h[i+(obj.Nx-3)] + h[i-(obj.Nx+1)])/2; # - (obj.aRad*obj.AdvecSpeed*T[i-(obj.Nx+1)]^4/2/pi)/2/obj.epsilon;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_B
                for j = 1:obj2.TNv
                    if obj2.points[j,2] > 0.0
                        g_wBC[i,j] = -obj.epsilon*(h[i-(obj.Nx)]+h[i+(obj.Nx-2)])/2 # - (obj.aRad*obj.AdvecSpeed*T[i+(obj.Nx-2)]^4/2/pi)/2/obj.epsilon;
                    end
                end
            end
        end

        for i in ghostidx_micro
            if i ∈ obj1.ghostidx_mic_L
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        g_wBC[i,j] = -g_wBC[i+1,j] - 2*obj.epsilon*(h[i-1]);
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_R
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        g_wBC[i,j] = -g_wBC[i-1,j] -2*obj.epsilon*(h[i-2]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_B
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        g_wBC[i,j] = -g_wBC[i+(2*obj.Nx-1),j] -2*obj.epsilon*(h[i+(obj.Nx-2)]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_T
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        g_wBC[i,j] = -g_wBC[i-(2*obj.Nx-1),j] -2*obj.epsilon*(h[i-(obj.Nx +1)]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            end
        end
    
    elseif obj.BCType == 2
        # Top and bottom boundaries are reflective, left and right are inflow.
        nT = [0.0,1.0];
        nB = [0.0,-1.0];
        for i in obj1.boundaryidx_micro
            if i ∈ obj1.boundaryidx_mic_L
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        g_wBC[i,j] = -obj.epsilon*(h[i-1]+h[i-2])/2;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_R
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        g_wBC[i,j] = -obj.epsilon*(h[i-2]+h[i-1])/2; # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_T
                for j = 1:obj2.TNv
                    if obj2.points[j,2] < 0.0
                        Omega = obj2.points[j,1:2];
                        Omega_ref = reflection(Omega,nT);
                        idx_list = findall(x -> abs(x - Omega_ref[1]) < 1e-10, obj2.points[:,1])
                        k = findall(x -> abs(x - Omega_ref[2]) < 1e-10,obj2.points[idx_list,2])[1]
                        idx = idx_list[k];
                        g_wBC[i,j] = g_wBC[i,idx];
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_B
                for j = 1:obj2.TNv
                    if obj2.points[j,2] > 0.0
                        Omega = obj2.points[j,1:2];
                        Omega_ref = reflection(Omega,nB);
                        idx_list = findall(x -> abs(x - Omega_ref[1]) < 1e-10, obj2.points[:,1])
                        k = findall(x -> abs(x - Omega_ref[2]) < 1e-10,obj2.points[idx_list,2])[1]
                        idx = idx_list[k];
                        g_wBC[i,j] = g_wBC[i,idx];
                    end
                end
            end
        end

        for i in ghostidx_micro
            if i ∈ obj1.ghostidx_mic_L
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        g_wBC[i,j] = -g_wBC[i+1,j] - 2*obj.epsilon*(h[i-1]);
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_R
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        g_wBC[i,j] = -g_wBC[i-1,j] -2*obj.epsilon*(h[i-2]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            end
        end
    else
        println("Not coded yet")
    end
    return g_wBC;
end

function BCg_lr(obj::Settings,obj1::StaggeredGrid, obj2::SNSystem,X,S,V,h,T)
    K = X*S;
    nT = [0.0,1.0];
    nB = [0.0,-1.0];
    ghostidx_micro = union(obj1.ghostidx_XMidY,obj1.ghostidx_XYMid)
    if obj.BCType == 0
         K[obj1.boundaryidx_micro,:] .= 0.0;
         K[ghostidx_micro,:] .= 0.0;
    elseif obj.BCType == 1 
        for i in obj1.boundaryidx_micro
            K_i = K[i,:];
            KV_i = K_i'*V';
            if i ∈ obj1.boundaryidx_mic_L
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        KV_i[j] = -obj.epsilon*(h[i-1]+h[i-2])/2 # - (obj.aRad*obj.AdvecSpeed*T[i-1]^4/2/pi)/2/obj.epsilon;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_R
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        KV_i[j] = -obj.epsilon*(h[i-2]+h[i-1])/2 # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_T
                for j = 1:obj2.TNv
                    if obj2.points[j,2] < 0.0
                        KV_i[j] = -obj.epsilon*(h[i+(obj.Nx-3)] + h[i-(obj.Nx+1)])/2; # - (obj.aRad*obj.AdvecSpeed*T[i-(obj.Nx+1)]^4/2/pi)/2/obj.epsilon;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_B
                for j = 1:obj2.TNv
                    if obj2.points[j,2] > 0.0
                        KV_i[j] = -obj.epsilon*(h[i-(obj.Nx)]+h[i+(obj.Nx-2)])/2 # - (obj.aRad*obj.AdvecSpeed*T[i+(obj.Nx-2)]^4/2/pi)/2/obj.epsilon;
                    end
                end
            end
            K[i,:] = KV_i*V;
        end

        for i in ghostidx_micro
            K_i = K[i,:];
            KV_i = K_i'*V';
            if i ∈ obj1.ghostidx_mic_L
                K_il = K[i+1,:];
                KV_il = K_il'*V';
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        KV_i[j] = -KV_il[j] - 2*obj.epsilon*(h[i-1]);
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_R
                K_ir = K[i-1,:];
                KV_ir = K_ir'*V';
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        KV_i[j] = -KV_ir[j] -2*obj.epsilon*(h[i-2]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_B
                K_ib = K[i+(2*obj.Nx-1),:];
                KV_ib = K_ib'*V';
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        KV_i[j] = -KV_ib[j] -2*obj.epsilon*(h[i+(obj.Nx-2)]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_T
                K_it = K[i-(2*obj.Nx-1),:];
                KV_it = K_it'*V';
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        KV_i[j] = -KV_it[j] -2*obj.epsilon*(h[i-(obj.Nx +1)]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            end
            K[i,:] = KV_i*V;
        end
    elseif obj.BCType == 2  
        for i in obj1.boundaryidx_micro
            K_i = K[i,:];
            KV_i = K_i'*V';
            if i ∈ obj1.boundaryidx_mic_L
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        KV_i[j] = -obj.epsilon*(h[i-1]+h[i-2])/2;
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_R
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        KV_i[j] = -obj.epsilon*(h[i-2]+h[i-1])/2; #
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_T
                for j = 1:obj2.TNv
                    if obj2.points[j,2] < 0.0
                        Omega = obj2.points[j,1:2];
                        Omega_ref = reflection(Omega,nT);
                        idx_list = findall(x -> abs(x - Omega_ref[1]) < 1e-10, obj2.points[:,1])
                        k = findall(x -> abs(x - Omega_ref[2]) < 1e-10,obj2.points[idx_list,2])[1]
                        idx = idx_list[k];
                        KV_i[j] = KV_i[idx];
                    end
                end
            elseif i ∈ obj1.boundaryidx_mic_B
                for j = 1:obj2.TNv
                    if obj2.points[j,2] > 0.0
                        Omega = obj2.points[j,1:2];
                        Omega_ref = reflection(Omega,nB);
                        idx_list = findall(x -> abs(x - Omega_ref[1]) < 1e-10, obj2.points[:,1])
                        k = findall(x -> abs(x - Omega_ref[2]) < 1e-10,obj2.points[idx_list,2])[1]
                        idx = idx_list[k];
                        KV_i[j] = KV_i[idx];
                    end
                end
            end
            K[i,:] = KV_i*V;
        end

        for i in ghostidx_micro
            K_i = K[i,:];
            KV_i = K_i'*V'; 
            if i ∈ obj1.ghostidx_mic_L
                K_ip = K[i+1,:];
                KV_ip = K_ip'*V';
                for j = 1:obj2.TNv
                    if obj2.points[j,1] > 0.0
                        KV_i[j] = -KV_ip[j] - 2*obj.epsilon*(h[i-1]);
                    end
                end
            elseif i ∈ obj1.ghostidx_mic_R
                K_im = K[i-1,:];
                KV_im = K_im'*V';
                for j = 1:obj2.TNv
                    if obj2.points[j,1] < 0.0
                        KV_i[j] = -KV_im[j] -2*obj.epsilon*(h[i-2]); # - (obj.aRad*obj.AdvecSpeed*T[i-2]^4/2/pi)/2/obj.epsilon; #
                    end
                end
            end
            K[i,:] = KV_i*V;
        end
    end
    X,S = py"qr"(K);
    return X,S,V;
end