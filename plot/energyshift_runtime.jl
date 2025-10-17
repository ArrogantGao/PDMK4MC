using CSV, DataFrames
using CairoMakie
using LsqFit

df_runtime = CSV.read("../data/dmk_energyshift_runtime.csv", DataFrame)

begin
    fig = Figure(size = (500, 400), fontsize = 20, dpi = 500)
    ax = Axis(fig[1, 1], xlabel = "N", ylabel = "time (s)", xscale = log10, yscale = log10)

    # scatter!(ax, df_runtime.n_src, df_runtime.time_pw, markersize = 12, marker = :circle, color = :green, label = "pw")
    # scatter!(ax, df_runtime.n_src, df_runtime.time_window, markersize = 12, marker = :utriangle, color = :red, label = "window")
    # scatter!(ax, df_runtime.n_src, df_runtime.time_diff, markersize = 12, marker = :diamond, color = :blue, label = "diff")
    # scatter!(ax, df_runtime.n_src, df_runtime.time_res, markersize = 12, marker = :utriangle, color = :red, label = "res")
    scatter!(ax, df_runtime.n_src, df_runtime.time_total, markersize = 12, marker = :diamond, color = :blue, label = "total")


    axislegend(ax, position = :rb)

    save("../figs/dmk_energyshift_runtime.png", fig)

    fig
end