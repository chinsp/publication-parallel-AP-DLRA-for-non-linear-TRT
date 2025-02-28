using Base: Float64
using PyCall
using Plots
using PyPlot
using DelimitedFiles
using WriteVTK


include("settings.jl")
include("solver.jl")

close("all")

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 30

cmap1 = plt.cm.get_cmap("jet");
cmap1.set_under("black");

Nx = 52;
Ny = 52;
problem = "Gaussian";
r = 30;

epsilon = 1e-5;
cfltype = "m"; 
s = Settings(Nx,Ny,r,epsilon,problem,cfltype);
solver = SolverMarshak(s);
sol_DL = SolveDiffusionLimit(solver);

epsilon_vals = [1.0,7.5e-1,5e-1,2.5e-1,1e-1,5e-2,1e-2,1e-3,1e-4];
cfltype = "m"

Temp_mat_FS_error = zeros(length(epsilon_vals));
Temp_mat_DL_error = zeros(length(epsilon_vals));
Temp_rad_FS_error = zeros(length(epsilon_vals));
Temp_rad_DL_error = zeros(length(epsilon_vals));


Full_profile = Dict();

fig1,ax1 = subplots(figsize=(10,10),dpi=100);

symbols = ["o-","v-","8-","s-","h-"]
i = 1;
j=1;
for epsilon in epsilon_vals[2:end]
    global i,cfltype,ax1,j;
    s.epsilon = epsilon;
    s.cfltype = cfltype;
    solver = SolverMarshak(s);
    println(i,",",s.epsilon,",")
    sol_parBUG = SolveParBUG(solver);
    sol_FS = SolveFull(solver);

    # save("2DMarshakNL/results/$problem/sol_FS_$i.jld2", "sol_FS", sol_FS)
    # save("2DMarshakNL/results/$problem/sol_ParBUG_$i.jld2", "sol_ParBUG", sol_parBUG)


    Temp_mat_FS_error[i] = norm(sol_FS["Temp"] - sol_DL["Temp"])/norm(sol_DL["Temp"]);
    Temp_mat_DL_error[i] = norm(sol_DL["Temp"] - sol_parBUG["Temp"])/norm(sol_DL["Temp"]);
    Temp_rad_FS_error[i] = norm((sol_FS["phi"]) - (sol_DL["phi"]))/norm((sol_DL["phi"]));
    Temp_rad_DL_error[i] = norm((sol_DL["phi"]) - (sol_parBUG["phi"]))/norm((sol_DL["phi"]));


    t = collect(range(0,s.tEnd,length(sol_parBUG["ranks"])));
    Full_profile["$i"]=sol_parBUG;

    if i == 1 
        ax1.plot(t,sol_parBUG["energy"],"o-", color = "blue",linestyle = "dotted", linewidth=4, markersize=10,alpha=0.8, markerfacecolor="none");
        ax1.plot(t,sol_FS["energy"],"o-", color = "red",linestyle="dashed", linewidth=4, markersize=10,alpha=0.8, markerfacecolor="none");
    elseif i == 9 
        ax1.plot(t,sol_parBUG["energy"], "x-", color = "blue",linestyle="dotted", linewidth=4, markersize=10,alpha=0.5, markerfacecolor="none");
        ax1.plot(t,sol_FS["energy"],"x-", color = "red",linestyle = "dashed", linewidth=4, markersize=10,alpha=0.5, markerfacecolor="none");
        j += 1;
    end
    i += 1;
end

ax1.set_ylabel("Energy");
ax1.set_xlabel(L"t");
fig1.canvas.draw();
legend_elements1 = [plt.Line2D([0], [0], marker = "o",markersize = 10,markerfacecolor="none", color="black", label=L"\varepsilon"*"= 1.0"),
                   plt.Line2D([0], [0],marker="x",markersize = 10, color="black", label=L"\varepsilon"*L"= 10^{-4}")]
legend_elements2 = [plt.Line2D([0], [0], linewidth = 4,linestyle = "dashed",alpha=0.8, color="red",  label="Full solver"),
                    plt.Line2D([0], [0], linewidth = 4,linestyle="dotted",alpha=0.8, color="blue", label="Parallel BUG solver")]
ax1.legend(handles=legend_elements1,fontsize=20);
fig1.legend(handles=legend_elements2,ncol=2,loc=9,fontsize=20);
fig1.savefig("2DMarshakNL/results/$problem/energy.pdf")


fig,ax = subplots(figsize=(10,10),dpi=100);
ax.semilogx(epsilon_vals,Temp_mat_DL_error,"-",color ="red",linestyle = "dashed",label = "Full solver",alpha=0.8, linewidth = 5, markersize = 10,markerfacecolor="none");
ax.semilogx(epsilon_vals,Temp_mat_FS_error,"-",color ="blue",linestyle="dotted",label = "Parallel BUG solver",alpha=0.8, linewidth = 5, markersize = 10,markerfacecolor="none");
ax.set_ylabel("Relative error",fontsize=25);
ax.set_xlabel(L"\varepsilon");
fig.canvas.draw();
fig.legend(ncol = 2,loc=9,fontsize=20);
plt.savefig("2DMarshakNL/results/$problem/AP_Temp_mat.pdf",bbox_inches="tight")