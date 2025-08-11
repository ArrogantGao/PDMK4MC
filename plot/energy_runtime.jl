using CSV, DataFrames
using CairoMakie
using LsqFit

df_dmk_energy_runtime = CSV.read("../data/dmk_energy_runtime.csv", DataFrame)

begin
    fig = Figure(size = (500, 400), fontsize = 20, dpi = 500)
    ax = Axis(fig[1, 1], xlabel = "N", ylabel = "time (s)", xscale = log10, yscale = log10)

    n_src = df_dmk_energy_runtime.n_src
    time_long = df_dmk_energy_runtime.time_planewave + df_dmk_energy_runtime.time_window + df_dmk_energy_runtime.time_difference
    time_short = df_dmk_energy_runtime.time_residual
    time_total = time_long + time_short

    scatter!(ax, n_src, time_long, markersize = 12, marker = :circle, color = :green, label = "long-range")
    scatter!(ax, n_src, time_short, markersize = 12, marker = :utriangle, color = :red, label = "short-range")
    scatter!(ax, n_src, time_total, markersize = 12, marker = :diamond, color = :blue, label = "total")

    @. model(x, p) = p[1] + x * p[2]
    fit = curve_fit(model, log10.(df_dmk_energy_runtime.n_src), log10.(df_dmk_energy_runtime.time_total), [1.0, 1.0])
    p = coef(fit)
    p_err = stderror(fit)

    xs = range(log10(1000 * 2), log10(1000 * 2^11), length = 100)
    lines!(ax, 10 .^ xs, 10 .^ model(xs, p), color = :blue, linewidth = 2, linestyle = :dash)

    axislegend(ax, position = :rb)

    save("../figs/dmk_energy_runtime.png", fig)

    fig
end