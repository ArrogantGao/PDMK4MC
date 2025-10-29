using CSV, DataFrames
using CairoMakie
using LsqFit
using Statistics

df_runtime = CSV.read("../data/dmk_energyshift_runtime.csv", DataFrame)

begin
    fig = Figure(size = (500, 400), fontsize = 20, dpi = 500)
    ax = Axis(fig[1, 1], xlabel = "N", ylabel = "time (μs)", xscale = log10)

    ns = unique(df_runtime.n_src)
    time_update = [mean(df_runtime[df_runtime.n_src .== n, :time_update]) for n in ns]
    time_shift = [mean(df_runtime[df_runtime.n_src .== n, :time_shift]) for n in ns]
    time_res = [mean(df_runtime[df_runtime.n_src .== n, :time_res]) for n in ns]

    scatter!(ax, ns, time_shift .* 1e6, markersize = 12, marker = :diamond, color = :blue, label = "propose")
    scatter!(ax, ns, time_res .* 1e6, markersize = 12, marker = :circle, color = :green, label = "residual")
    scatter!(ax, ns, time_update .* 1e6, markersize = 12, marker = :utriangle, color = :red, label = "agree")

    ylims!(ax, 0, 500)
    axislegend(ax, position = :lt)

    save("../figs/dmk_energyshift_runtime.png", fig)

    fig
end