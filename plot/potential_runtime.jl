using CSV, DataFrames
using CairoMakie
using LsqFit

df_accuracy = CSV.read("../data/potential_accuracy.csv", DataFrame)
df_runtime = CSV.read("../data/dmk_potential_runtime.csv", DataFrame)
df_ewald_runtime = CSV.read("../data/ewald_potential_runtime.csv", DataFrame)

begin
    fig = Figure(size = (1000, 400), fontsize = 20)
    ax_1 = Axis(fig[1, 1], xlabel = "N", ylabel = "absolute error", xscale = log10, yscale = log10)
    ax_2 = Axis(fig[1, 2], xlabel = "N", ylabel = "time (s)", xscale = log10)


    scatter!(ax_1, df_accuracy.n_src, df_accuracy.abserr_dmk, markersize = 12, marker = :diamond, color = :blue, label = "DMK")
    scatter!(ax_1, df_accuracy.n_src, df_accuracy.abserr_ewald, markersize = 12, marker = :utriangle, color = :red, label = "Ewald")

    # xlims!(ax_1, 10^(3.8), 10^(5.2))
    ylims!(ax_1, 1e-5, 1e-1)

    @. model(x, p) = p[1] + x * p[2]
    fit = curve_fit(model, log10.(df_runtime.n_src), df_runtime.time_total, [1.0, 1.0])
    p = coef(fit)
    p_err = stderror(fit)

    @. model_ewald(x, p) = p[1] * x + p[2]
    fit_ewald = curve_fit(model_ewald, log10.(df_ewald_runtime.n_src), log10.(df_ewald_runtime.time_total), [1.0, 1.0])
    p_ewald = coef(fit_ewald)
    p_ewald_err = stderror(fit_ewald)

    ns = 10 .^ range(3.8, 8.2, length = 100)

    scatter!(ax_2, df_runtime.n_src, df_runtime.time_total, markersize = 12, marker = :diamond, color = :blue)
    lines!(ax_2, ns, model(log10.(ns), p), color = :blue, linewidth = 2, linestyle = :dash)

    scatter!(ax_2, df_ewald_runtime.n_src, df_ewald_runtime.time_total, markersize = 12, marker = :utriangle, color = :red)
    lines!(ax_2, ns, 10.0 .^ model_ewald(log10.(ns), p_ewald), color = :red, linewidth = 2, linestyle = :dash)

    axislegend(ax_1, position = :rb)

    xlims!(ax_2, 10^(3.8), 10^(8.2))
    ylims!(ax_2, -0.002, 0.025)

    save("../figs/potential_accuracy_runtime.png", fig)

    fig
end