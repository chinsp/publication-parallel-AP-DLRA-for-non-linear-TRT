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


Nx = 52;
Ny = 52;
problem = "MarshakWave";
r = 30;


epsilon = 1.0;
cfltype = "m"; 
s = Settings(Nx,Ny,r,epsilon,problem,cfltype);
solver = SolverMarshak(s);

tEnd_list = [0.01 * 1e-10,0.02 * 1e-10,0.03 * 1e-10,0.04 * 1e-10,0.05 * 1e-10]; # 
symbols = ["o-","v-","8-","s-","h-"]
close("all");
fig1,ax1 = subplots(figsize=(10,10),dpi=100);
fig2,ax2 = subplots(figsize=(10,10),dpi=100);
fig3,ax3 = subplots(figsize=(10,10),dpi=100);

i = 1;
for tEnd in tEnd_list
    global i, ax1,ax2, ax3;
    local fig,ax
    s.tEnd = tEnd;
    solver = SolverMarshak(s);

    sol_FS = SolveFull(solver);
    sol_ParBUG = SolveParBUG(solver);

    T_plot3 = sol_FS["Temp"][solver.SG.idx_XMidYMid];
    T_plotMat3 = reshape(T_plot3,Nx-1,Ny-1);

    phi_plot3 = sol_FS["phi"][solver.SG.idx_XMidYMid];
    phi_plotMat3 = reshape(phi_plot3,Nx-1,Ny-1);

    T_plot2 = sol_ParBUG["Temp"][solver.SG.idx_XMidYMid];
    T_plotMat2 = reshape(T_plot2,Nx-1,Ny-1);

    phi_plot2 = sol_ParBUG["phi"][solver.SG.idx_XMidYMid];
    phi_plotMat2 = reshape(phi_plot2,Nx-1,Ny-1);
    
    y0 = 26;

    ax1.plot(s.xMid.*1000,T_plotMat3'[y0,:]./11604,symbols[i],color = "red",alpha = 0.8, linewidth=4,linestyle= "dashed", markersize=10,markerfacecolor="none");
    ax1.plot(s.xMid.*1000,T_plotMat2'[y0,:]./11604,symbols[i],color = "blue",alpha = 0.8, linewidth=4,linestyle= "dotted", markersize=10,markerfacecolor="none");

    ax2.plot(s.xMid.*1000,((2*pi .*phi_plotMat3'/s.aRad/s.AdvecSpeed).^(1/4))[y0,:]./11604,symbols[i],color = "red",alpha = 0.8, linewidth=4,linestyle= "dashed", markersize=10,markerfacecolor="none");
    ax2.plot(s.xMid.*1000,((2*pi .*phi_plotMat2'/s.aRad/s.AdvecSpeed).^(1/4))[y0,:]./11604,symbols[i],color = "blue", alpha = 0.8, linewidth=4,linestyle= "dotted", markersize=10,markerfacecolor="none");

    ax3.plot(s.xMid.*1000,(phi_plotMat3')[y0,:]./11604,symbols[i],color = "red",alpha = 0.8, linewidth=4,linestyle= "dashed", markersize=10,markerfacecolor="none");
    ax3.plot(s.xMid.*1000,(phi_plotMat2')[y0,:]./11604,symbols[i],color = "blue",alpha = 0.8, linewidth=5,linestyle= "dotted", markersize=10,markerfacecolor="none");

    # Save solutions
    # Recommended
    # save("2DMarshakNL/results/$problem/sol_FS_$i.jld2", "sol_FS", sol_FS)

    # save("2DMarshakNL/results/$problem/sol_ParBUG_$i.jld2", "sol_ParBUG", sol_ParBUG)

    i += 1;

end

ax1.set_ylabel("Material Temperature");
ax1.set_xlabel(L"x");
fig1.canvas.draw();
legend_elements1 = [plt.Line2D([0], [0], marker = "o",markersize = 10,markerfacecolor="none", color="black", label=L"1"*"ps"),
                   plt.Line2D([0], [0],marker="v",markersize = 10,markerfacecolor="none", color="black", label=L"2"*"ps"),
                   plt.Line2D([0], [0],marker="8",markersize = 10,markerfacecolor="none", color="black", label=L"3"*"ps"),
                   plt.Line2D([0], [0],marker="s",markersize = 10,markerfacecolor="none", color="black", label=L"4"*"ps"),
                   plt.Line2D([0], [0],marker="h",markersize = 10,markerfacecolor="none", color="black", label=L"5"*"ps")]
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")];
ax1.legend(handles=legend_elements1,fontsize=20);
fig1.legend(handles=legend_elements2,ncol=2,fontsize=20,loc=9);
fig1.savefig("2DMarshakNL/results/$problem/cross_section_temp.pdf",bbox_inches="tight")  ;

ax2.set_ylabel("Radiation temperature");
ax2.set_xlabel(L"x");
fig2.canvas.draw();
legend_elements1 = [plt.Line2D([0], [0], marker = "o",markersize = 10,markerfacecolor="none", color="black", label=L"1"*"ps"),
                   plt.Line2D([0], [0],marker="v",markersize = 10,markerfacecolor="none", color="black", label=L"2"*"ps"),
                   plt.Line2D([0], [0],marker="8",markersize = 10,markerfacecolor="none", color="black", label=L"3"*"ps"),
                   plt.Line2D([0], [0],marker="s",markersize = 10,markerfacecolor="none", color="black", label=L"4"*"ps"),
                   plt.Line2D([0], [0],marker="h",markersize = 10,markerfacecolor="none", color="black", label=L"5"*"ps")]
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")];
ax2.legend(handles=legend_elements1,fontsize=20);
fig2.legend(handles=legend_elements2,ncol=2,fontsize=20,loc=9);
fig2.savefig("2DMarshakNL/results/$problem/cross_section_radTemp.pdf",bbox_inches="tight")

ax3.set_ylabel("Scalar flux");
ax3.set_xlabel(L"x");
fig3.canvas.draw();
legend_elements1 = [plt.Line2D([0], [0], marker = "o",markersize = 10,markerfacecolor="none", color="black", label=L"1"*"ps"),
                   plt.Line2D([0], [0],marker="v",markersize = 10,markerfacecolor="none", color="black", label=L"2"*"ps"),
                   plt.Line2D([0], [0],marker="8",markersize = 10,markerfacecolor="none", color="black", label=L"3"*"ps"),
                   plt.Line2D([0], [0],marker="s",markersize = 10,markerfacecolor="none", color="black", label=L"4"*"ps"),
                   plt.Line2D([0], [0],marker="h",markersize = 10,markerfacecolor="none", color="black", label=L"5"*"ps")]
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")];
ax3.legend(handles=legend_elements1,fontsize=20);
fig3.legend(handles=legend_elements2,ncol=2,fontsize=20,loc=9);
fig3.savefig("2DMarshakNL/results/$problem/cross_section_flux.pdf",bbox_inches="tight");


epsilon = 1e-4;
cfltype = "m"; # In hyperbolic regime set to "h"; in diffusive regime "p"; in a mixed regime "m"
s = Settings(Nx,Ny,r,epsilon,problem,cfltype);
solver = SolverMarshak(s);
sol_DL = SolveDiffusionLimit(solver);

T_plot4 = sol_DL["Temp"][solver.SG.idx_XMidYMid];
T_plotMat4 = reshape(T_plot4,Nx-1,Ny-1);

close("all")
fig = figure("T",figsize=(10,10),dpi=100)
ax = gca()
im2 = pcolormesh(s.xMid,s.yMid,T_plotMat4'./11604,vmin=0.0,shading="auto",cmap = cmap1,rasterized=true)
color_bar = fig.colorbar(im2, ax=ax, pad=0.03, shrink = 0.71)
ax.tick_params("both",labelsize=20)
ax.set(adjustable = "box", aspect = "equal")
plt.axis("off");
plt.savefig("2DMarshakNL/results/$problem/DL_Temp.pdf",bbox_inches="tight")

sol_FS = SolveFull(solver);
# save("2DMarshakNL/results/$problem/sol_FS_DL.jld2", "sol_FS", sol_FS)
sol_ParBUG = SolveParBUG(solver);
# save("2DMarshakNL/results/$problem/sol_ParBUG_DL.jld2", "sol_ParBUG", sol_ParBUG)

T_plot3 = sol_FS["Temp"][solver.SG.idx_XMidYMid];
T_plotMat3 = reshape(T_plot3,Nx-1,Ny-1);

phi_plot3 = sol_FS["phi"][solver.SG.idx_XMidYMid];
phi_plotMat3 = reshape(phi_plot3,Nx-1,Ny-1);

T_plot2 = sol_ParBUG["Temp"][solver.SG.idx_XMidYMid];
T_plotMat2 = reshape(T_plot2,Nx-1,Ny-1);

phi_plot2 = sol_ParBUG["phi"][solver.SG.idx_XMidYMid];
phi_plotMat2 = reshape(phi_plot2,Nx-1,Ny-1);

close("all")
fig1,ax1 = subplots(figsize=(10,10),dpi=100);
fig2,ax2 = subplots(figsize=(10,10),dpi=100);
fig3,ax3 = subplots(figsize=(10,10),dpi=100);

ax1.plot(s.xMid.*1000,T_plotMat4'[26,:]./11604,"-",color = "black", label = "Rosseland", linewidth=3, markersize=10);
ax2.plot(s.xMid.*1000,T_plotMat4'[26,:]./11604,"-",color = "black", label = "Rosseland", linewidth=3, markersize=10);
ax3.plot(s.xMid.*1000,(s.aRad*s.AdvecSpeed .*(T_plotMat4').^4 ./2/pi)[26,:]./11604,"-",color = "black", label = "Rosseland", linewidth=3, markersize=10);

ax1.plot(s.xMid.*1000,T_plotMat3'[26,:]./11604,"-",linestyle = "dashed",color = "red",alpha = 0.8, linewidth=6, markersize=10);
ax1.plot(s.xMid.*1000,T_plotMat2'[26,:]./11604,"-",linestyle="dotted",color = "blue", alpha = 0.8, linewidth=6, markersize=10);

ax2.plot(s.xMid.*1000,((2*pi .*phi_plotMat3'/s.aRad/s.AdvecSpeed).^(1/4))[26,:]./11604,"-",linestyle = "dashed",color = "red",alpha = 0.8, linewidth=6, markersize=10);
ax2.plot(s.xMid.*1000,((2*pi .*phi_plotMat2'/s.aRad/s.AdvecSpeed).^(1/4))[26,:]./11604,"-",linestyle="dotted",color = "blue",alpha = 0.8, linewidth=6, markersize=10);

ax3.plot(s.xMid.*1000,phi_plotMat3'[26,:]./11604,"-",linestyle = "dashed",color = "red",alpha = 0.8, linewidth=6, markersize=10);
ax3.plot(s.xMid.*1000,phi_plotMat2'[26,:]./11604,"-",linestyle="dotted",color = "blue", alpha = 0.8, linewidth=6, markersize=10);

ax1.set_ylabel("Material temperature");
ax1.set_xlabel(L"x");
fig1.canvas.draw();
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,alpha=0.8, color="black",  label="Rosseland"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")]
fig1.legend(handles=legend_elements2,ncol=3,fontsize=20,loc=9);
fig1.savefig("2DMarshakNL/results/$problem/cross_section_temp_diff_lim.pdf",bbox_inches="tight")  

ax2.set_ylabel("Radiation temperature");
ax2.set_xlabel(L"x");
fig2.canvas.draw();
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,alpha=0.8, color="black",  label="Rosseland"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")]
fig2.legend(handles=legend_elements2,ncol=3,fontsize=20,loc=9);
fig2.savefig("2DMarshakNL/results/$problem/cross_section_radTemp_diff_lim.pdf",bbox_inches="tight")

ax3.set_ylabel("Scalar flux");
ax3.set_xlabel(L"x");
fig3.canvas.draw();
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,alpha=0.8, color="black",  label="Rosseland"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")]
fig3.legend(handles=legend_elements2,ncol=3,fontsize=20,loc=9);
fig3.savefig("2DMarshakNL/results/$problem/cross_section_flux_diff_lim.pdf",bbox_inches="tight")
