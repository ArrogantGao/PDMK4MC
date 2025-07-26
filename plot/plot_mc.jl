using CSV, DataFrames
using CairoMakie
using LsqFit

df_accuracy = CSV.read("data/mc_accuracy.csv", DataFrame)
df_runtime = CSV.read("data/mc_runtime_old.csv", DataFrame)
df_ewald_runtime = CSV.read("data/ewald_runtime.csv", DataFrame)

begin
    fig = Figure(size = (1000, 400), fontsize = 16)
    ax_1 = Axis(fig[1, 1], xlabel = "N", ylabel = "absolute error", xscale = log10, yscale = log10)
    ax_2 = Axis(fig[1, 2], xlabel = "N", ylabel = "time (s)", xscale = log10)


    scatter!(ax_1, df_accuracy.n_src, df_accuracy.abserr_dmk, markersize = 12, marker = :diamond, color = :blue)
    scatter!(ax_1, df_accuracy.n_src, df_accuracy.abserr_ewald, markersize = 12, marker = :triangle, color = :red)

    xlims!(ax_1, 10^(3.8), 10^(5.2))
    ylims!(ax_1, 1e-5, 1e-1)

    @. model(x, p) = p[1] + x * p[2]
    fit = curve_fit(model, log10.(df_runtime.n_src), df_runtime.time_total, [1.0, 1.0])
    p = coef(fit)
    p_err = stderror(fit)

    scatter!(ax_2, df_runtime.n_src, df_runtime.time_total, markersize = 12, marker = :diamond, color = :blue)
    lines!(ax_2, df_runtime.n_src, model(log10.(df_runtime.n_src), p), color = :blue, linewidth = 2, linestyle = :dash)

    scatter!(ax_2, df_ewald_runtime.n_src, df_ewald_runtime.time_total, markersize = 12, marker = :triangle, color = :red)

    save("figs/mc_accuracy.pdf", fig)

    fig
end