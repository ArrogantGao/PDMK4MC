using CSV, DataFrames
using CairoMakie
using LsqFit

df_runtime = CSV.read("../data/dmk_energyshift_runtime.csv", DataFrame)

begin
    fig = Figure(size = (500, 400), fontsize = 20, dpi = 500)
    ax = Axis(fig[1, 1], xlabel = "N", ylabel = "time (μs)", xscale = log10)

    scatter!(ax, df_runtime.n_src, df_runtime.time_update .* 1e6, markersize = 12, marker = :utriangle, color = :red, label = "agree")
    scatter!(ax, df_runtime.n_src, df_runtime.time_shift .* 1e6, markersize = 12, marker = :diamond, color = :blue, label = "propose")

    ylims!(ax, 0, 600)
    axislegend(ax, position = :lt)

    save("../figs/dmk_energyshift_runtime.png", fig)

    fig
end