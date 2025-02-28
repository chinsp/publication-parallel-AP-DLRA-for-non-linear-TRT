using Base: Float64
using PyCall
using Plots
using PyPlot
using DelimitedFiles
using WriteVTK

using JLD2
include("settings.jl")
include("solver.jl")

cmap1 = plt.cm.get_cmap("jet");
cmap1.set_under("black");

close("all")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 30

Nx = 102;
Ny = 102;
problem = "Hohlraum";
r = 30;

epsilon = 1.0;
cfltype = "m"; 
s = Settings(Nx,Ny,r,epsilon,problem,cfltype);
solver = SolverMarshak(s);

sol_FS = SolveFull(solver);

T_plot3 = sol_FS["Temp"][solver.SG.idx_XMidYMid];
T_plotMat3 = reshape(T_plot3,Nx-1,Ny-1);

phi_plot3 = sol_FS["phi"][solver.SG.idx_XMidYMid];
phi_plotMat3 = reshape(phi_plot3,Nx-1,Ny-1);

close("all");
fig = figure("T",figsize=(10,10),dpi=100);
ax = gca();
im2 = plt.pcolormesh(s.xMid,s.yMid,T_plotMat3',shading="auto",cmap=cmap1,vmin = 0.0,vmax=1.0,rasterized=true);
color_bar = fig.colorbar(im2, ax=ax, pad=0.03, shrink = 0.71)
ax.tick_params("both",labelsize=20);
ax.set(adjustable = "box", aspect = "equal");
plt.axis("off");
plt.savefig("2DMarshakNL/results/$problem/FS_Temp.pdf",bbox_inches="tight");


close("all")
fig = figure("T",figsize=(10,10),dpi=100)
ax = gca();
im2 = pcolormesh(s.xMid,s.yMid,(2*pi .* phi_plotMat3' ./s.aRad/s.AdvecSpeed).^(1/4),shading="auto",cmap = cmap1,vmin=0.0,vmax=1.0,rasterized=true)
color_bar = fig.colorbar(im2, ax=ax, pad=0.03, shrink = 0.71)
ax.tick_params("both",labelsize=20)
ax.set(adjustable = "box", aspect = "equal")
plt.axis("off");
plt.savefig("2DMarshakNL/results/$problem/FS_RadTep.pdf",bbox_inches="tight")

sol_ParBUG = SolveParBUG(solver);

T_plot2 = sol_ParBUG["Temp"][solver.SG.idx_XMidYMid];
T_plotMat2 = reshape(T_plot2,Nx-1,Ny-1);

phi_plot2 = sol_ParBUG["phi"][solver.SG.idx_XMidYMid];
phi_plotMat2 = reshape(phi_plot2,Nx-1,Ny-1);

close("all");
fig = figure("T",figsize=(10,10),dpi=100);
ax = gca();
im2 = plt.pcolormesh(s.xMid,s.yMid,T_plotMat2',shading="auto",cmap=cmap1,vmin = 0.0,vmax=1.0,rasterized=true);
color_bar = fig.colorbar(im2, ax=ax, pad=0.03, shrink = 0.71)
ax.tick_params("both",labelsize=20);
ax.set(adjustable = "box", aspect = "equal");
plt.axis("off");
plt.savefig("2DMarshakNL/results/$problem/ParBUG_Temp.pdf",bbox_inches="tight");


close("all")
fig = figure("T",figsize=(10,10),dpi=100)
ax = gca();
im2 = pcolormesh(s.xMid,s.yMid,(2*pi .* phi_plotMat2'./s.aRad/s.AdvecSpeed).^(1/4),shading="auto",cmap = cmap1,vmin=0.0,vmax=1.0, rasterized=true)
color_bar = fig.colorbar(im2, ax=ax, pad=0.03, shrink = 0.71)
ax.tick_params("both",labelsize=20)
ax.set(adjustable = "box", aspect = "equal")
plt.axis("off");
plt.savefig("2DMarshakNL/results/$problem/ParBUG_RadTemp.pdf",bbox_inches="tight")

t = collect(range(0,s.tEnd,length(sol_ParBUG["ranks"])));


fig,ax = subplots(figsize=(10,10),dpi=100);
ax.plot(t[1:end],sol_ParBUG["ranks"],"-",color = "green",linewidth=2);
ax.set_xlim([0,s.tEnd]);
ax.set_ylabel("rank");
ax.set_xlabel(L"t");
fig.canvas.draw();
plt.savefig("2DMarshakNL/results/$problem/test_ranks.pdf",bbox_inches="tight")
