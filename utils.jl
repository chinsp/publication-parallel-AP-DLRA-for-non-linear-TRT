function Vec2Mat(nx,ny,v)
    m = zeros(nx,ny);
    for i = 1:nx
        for j = 1:ny
            m[i,j] = v[(i-1)*ny + j]
        end
    end
    return m;
end

function Mat2Vec(mat)
    nx = size(mat,1)
    ny = size(mat,2)
    m = size(mat,3)
    v = zeros(nx*ny,m);
    for i = 1:nx
        for j = 1:ny
            v[(i-1)*ny + j,:] = mat[i,j,:]
        end
    end
    return v;
end

function Ten2Vec(ten)
    nx = size(ten,1)
    ny = size(ten,2)
    nxi = size(ten,3)
    m = size(ten,4)
    v = zeros(nx*ny,m*nxi);
    for i = 1:nx
        for j = 1:ny
            for l = 1:nxi
                for k = 1:m
                    v[(i-1)*ny + j,(l-1)*m .+ k] = ten[i,j,l,k]
                end
            end
        end
    end
    return v;
end

function vectorIndex(ny,i,j)
    return (i-1)*ny + j;
end

function reflection(Omega,n)
    return Omega - 2*dot(Omega,n)*n;
end