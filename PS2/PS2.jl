using LinearAlgebra, SparseArrays
using ProgressMeter
using Plots
using LaTeXStrings
Plots.pyplot()

pressure(r, ru, rv, rE, γ=1.4) = @. (γ - 1) * (rE - (ru^2 + rv^2)/2r)

function euler_fluxes(r, ru, rv, rE)
    """
    Return the x any y components of the fluxes of the solution components
    Matrix F in (4) in PS2

    Arguments:
        r, ru, rv, rE: Solution components
    Returns:
        Frx, Fry, Frux, Fruy, Frvx, Frvy, FrEx, FrEy: flux-componets
    """
    p = pressure(r, ru, rv, rE)
    Frx  = ru
    Fry  = rv
    Frux = @. ru^2 / r + p
    Fruy = @. ru * rv / r
    Frvx = @. ru * rv / r
    Frvy = @. rv^2 / r + p
    FrEx = @. ru / r * (rE + p)
    FrEy = @. rv / r * (rE + p)
    return Frx, Fry, Frux, Fruy, Frvx, Frvy, FrEx, FrEy
end

function pade_x(f, h, mats)
    """
    Calculates derivative in x-direction (lines) of nxn square matrix
    using Padé-scheme with periodic boundaries

    Arguments:
        f: square matrix to calculate line-derivaties of
        h: spatial steplength
        mats: pre-generated matricies performing
              derivative according to equation in part b
    Returns:
        df: square matrix
    """
    lhs, rhs = mats
    srhs = rhs * 3 / h  # scale with correct params

    df = lhs \ (srhs * f)
    return df
end

function compact_div(Fx, Fy, h, mats)
    """
    Computes the divergence of a field F=[Fx, Fy] using the 1d padé-scheme
    Uses pade_x to compute xdiv on Fx
    transposes Fy to use pade_x to get ydiv on Fy, retransposes to get correct result
    returns sum

    Arguments:
        Fx, Fy: x and y part of vector-field F to compute flux of
        h: spatial steplength
        mats: pre-computed matrices, see doc of pade_x
    Returns:
        divF: divergence of field F
    """
    dFx = pade_x(Fx, h, mats)
    dFy = pade_x(Matrix(Fy'), h, mats)
    return dFx .+ Matrix(dFy')
end

function compact_div(Fx, Fy, h)
    """
    Wrapper for autograder
    makes the matricies needed for my code,
    and passes them on to my function
    Not tested, on its own.
    """
    n = size(Fx)[1]
    mats = make_matricies(n, 0)
    compact_div(Fx, Fy, h, mats[1])
end

function filter_x(u, filters)
    """
    Filters square matrix u in x-direction
    Using equation in part c

    Arguments:
        u: square matrix to filter in x-dir
        filters: pre-computed matrices, same method as for pade_x
    Returns:
        f: filtered u
    """
    lhs, rhs = filters

    f = lhs \ (rhs * u)
    return f
end

function my_compact_filter(u, filters)
    """
    Filters square matrix u in x and y direction
    Uses same trick as divergence by filtering in x,
    then transposing to filter in y, then retrenspose

    Arguments:
        u: square matrix to filter
        filters: pre-computed filter matricies, see doc of filter_x
    Returns:
        fu: filtered u
    """
    u = filter_x(u, filters)  # filter in x-direction
    u = filter_x(Matrix(u'), filters) # filter in y-direction
    return Matrix(u')
end

function compact_filter(u, α)
    """
    wrapper for my_compact_filter
    Not tested
    """
    n = size(u)[1]
    mats = make_matricies(n, α)
    my_compact_filter(u, mats[2])
end

function euler_rhs(r, ru, rv, rE, h, mats)
    """
    Rhs of differential equation du_dt = -divF(u)
    Calculates fluxes of each component,
    calculates divergence of each component using fluxes
    returns divergence of solution
    """
    Frx, Fry, Frux, Fruy, Frvx, Frvy, FrEx, FrEy = euler_fluxes(r, ru, rv, rE)
    fr  = compact_div(Frx , Fry , h, mats)
    fru = compact_div(Frux, Fruy, h, mats)
    frv = compact_div(Frvx, Frvy, h, mats)
    frE = compact_div(FrEx, FrEy, h, mats)
    return -fr, -fru, -frv, -frE
end

function euler_rhs(r, ru, rv, rE, h)
    """
    wrapper for autograder compatability
    Not tested
    """
    mats = make_matricies(size(r)[1], 0)
    euler_rhs(r, ru, rv, rE, h, mats)
end

function my_euler_rk4step(r, ru, rv, rE, h, k, mats)
    """
    Takes a single rk4-step of solution
    filters solution before returning

    Arguments
        r, ru, rv, rE: components of solution, each element is a nxn matrix
        h: spatial stepsize
        k: temporal steplength
        mats: pre-computed matricies for derivating and filtering
    Returns:
        Next step of r, ru, rv, rE
    """
    pade, filters = mats  # divergence and filter matricies
    k = 0.5 * k           # temporal stepsize, halved for convenience
    # standard sub-steps of rk4
    r1, ru1, rv1, rE1 = euler_rhs(r, ru, rv, rE, h, pade)
    r2, ru2, rv2, rE2 = euler_rhs(r+ k*r1, ru+ k*ru1, rv+ k*rv1, rE+ k*rE1, h, pade)
    r3, ru3, rv3, rE3 = euler_rhs(r+ k*r2, ru+ k*ru2, rv+ k*rv2, rE+ k*rE2, h, pade)
    r4, ru4, rv4, rE4 = euler_rhs(r+2k*r3, ru+2k*ru3, rv+2k*rv3, rE+2k*rE3, h, pade)

    # update each component
    r  += k/3 * (r1  + 2r2  + 2r3  + r4 )
    ru += k/3 * (ru1 + 2ru2 + 2ru3 + ru4)
    rv += k/3 * (rv1 + 2rv2 + 2rv3 + rv4)
    rE += k/3 * (rE1 + 2rE2 + 2rE3 + rE4)

    # filter each component
    r  = my_compact_filter(r , filters)
    ru = my_compact_filter(ru, filters)
    rv = my_compact_filter(rv, filters)
    rE = my_compact_filter(rE, filters)

    return r, ru, rv, rE
end

function euler_rk4step(r, ru, rv, rE, h, k, α)
    """
    wrapper for euler_rk4step
    Not tested
    """
    mats = make_matricies(size(r)[1], α)
    euler_rk4step(r, ru, rv, rE, h, k, mats)
end

function euler_vortex(x, y, time, pars)
    """
    Test-function to generate exact solution
    """
    γ  = 1.4
    rc = pars[1]
    ϵ  = pars[2]
    M₀ = pars[3]
    θ  = pars[4]
    x₀ = pars[5]
    y₀ = pars[6]

    r∞ = 1
    u∞ = 1
    E∞ = 1/(γ*M₀^2*(γ - 1)) + 1/2
    p∞ = (γ - 1) * (E∞ - 1/2)
    ubar = u∞ * cos(θ)
    vbar = u∞ * sin(θ)
    f = @. (1 - ((x - x₀) - ubar*time)^2 - ((y - y₀) - vbar*time)^2) / rc^2

    u = @. u∞ * (cos(θ) - ϵ*((y - y₀)-vbar*time) / (2π*rc) * exp(f/2))
    v = @. u∞ * (sin(θ) + ϵ*((x - x₀)-ubar*time) / (2π*rc) * exp(f/2))
    r = @. r∞ * (1 - ϵ^2 * (γ - 1) * M₀^2/(8π^2) * exp(f))^(1/(γ-1))
    p = @. p∞ * (1 - ϵ^2 * (γ - 1) * M₀^2/(8π^2) * exp(f))^(γ/(γ-1))

    ru = @. r*u
    rv = @. r*v
    rE = @. p/(γ - 1) + 1/2 * (ru^2 + rv^2) / r

    r, ru, rv, rE
end

# utility-funcs
error(x, y) = maximum(abs.(x .- y))
lerror(x, y) = log(error(x, y))

function make_matricies(n, α)
    """
    Pre-compute matricies that differentiate and filter in x-direction
    Equations in part b and c

    Arguments:
        n: number of points in discretisation
        α: filtering-parameter
    Returns:
        mats:
            mats[1,1] = lhs of differentiation
            mats[1,2] = rhs of differentiation
            mats[2,1] = lhs of filtering
            mats[2,2] = rhs of filtering
    """
    # filter-params
    a = (5. / 8 + 3. * α / 4) * ones(n)
    b(n) = (α + 0.5) * ones(n) / 2
    c(n) = (α / 4. - 1/8) * ones(n) / 2

    filter_rhs = spdiagm(-n+1=>b(1), -n+2=>c(2), -2=>c(n-2), -1=>b(n-1), 0=>a, 1=>b(n-1), 2=>c(n-2), n-2=>c(2), n-1=>b(1))
    filter_lhs = spdiagm(-n+1=>[α], -1=>α*ones(n-1), 0=>ones(n), 1=>α*ones(n-1), n-1=>[α])

    div_lhs = spdiagm(-n+1=>[1], -1=>ones(n-1), 0=>4ones(n), 1=>ones(n-1), n-1=>[1])
    div_rhs = spdiagm(-n+1=>[1], -1=>-ones(n-1), 1=>ones(n-1), n-1=>[-1])

    # pre-compute LU-factorisation, to significantly speed-up (~6x) solving by backslash
    div_LU = lu(div_lhs)
    filter_LU = lu(filter_lhs)

    return [[div_LU, div_rhs], [filter_LU, filter_rhs]]
end


function animate_vortex(n=32, α=0.499)
    """
    Animates density and energy of computed and analytic euler vortex
    Takes forever
    """
    default(legend=false)
    h = 10 / n
    T = 5sqrt(2)
    m = Int(ceil(10T / 3h))
    k = T / m

    s = h:h:10
    x = repeat(s, 1, length(s))
    y = Matrix(x')

    pars = [0.5, 1, 0.5, π/4, 2.5, 2.5]
    mats = make_matricies(n, α)

    r, ru, rv, rE = euler_vortex(x, y, 0, pars)

    l = @layout [a b; c d]
    p = plot(layout=l)

    pbar = Progress(m; showspeed=true)
    @gif for i in 1:m
        r, ru, rv, rE = my_euler_rk4step(r, ru, rv, rE, h, k, mats)
        r0, ru0, rv0, rE0 = euler_vortex(x, y, i*k, pars)

        plot!(p[1], x, y, r ,      st=:contourf, label=false, title=L"\rho")
        plot!(p[2], x, y, r0,      st=:contourf, label=false, title="true "*L"\rho")
        plot!(p[3], x, y, rE ./r , st=:contourf, label=false, title="E")
        plot!(p[4], x, y, rE0./r0, st=:contourf, label=false, title="true E")
        ProgressMeter.next!(pbar; showvalues=[(:total_iterations, m), (:current_iteration, i), (:time, (round(i*k, digits=5)))])
    end every 5
end

function simple_vortex(n=32, α=0.499)
    """
    Test-function running a vortex and returning error in density function
    """
    h = 10 / n               # spatial steplength
    T = 5sqrt(2)             # Total time
    m = Int(ceil(10T / 3h))  # points in temporal discretisation
    k = T / m                # temporal steplength

    s = h:h:10
    x = repeat(s, 1, length(s))
    y = Matrix(x')

    mats = make_matricies(n, α)

    pars = [0.5, 1, 0.5, π/4, 2.5, 2.5]
    # True solution at time T
    r0, ru0, rv0, rE0 = euler_vortex(x, y, T, pars)

    # initial condition
    r, ru, rv, rE = euler_vortex(x, y, 0, pars)

    pbar = Progress(m; showspeed=true)
    for i in 1:m
        r, ru, rv, rE = my_euler_rk4step(r, ru, rv, rE, h, k, mats)
        ProgressMeter.next!(pbar; showvalues=[(:total_iterations, m), (:current_iteration, i), (:time, (round(i*k, digits=5)))])
    end
    error(r, r0)
end

function votrex_convergence(αs, ns)
    """
    For a set of filtering-params and discretisation-points,
    calculate max-norm error of all components of solution at final time
    return list of step-length and error for each.
    Used by vortex_convergence(αs, ns, data) to produce convergence plot
    """
    T = 5sqrt(2)
    out = []
    for α in αs
        errs = []
        hs = []

        for n in ns
            println("α=$(α), n=$(n)")
            h = 10 / n
            m = Int(ceil(10T / 3h))
            k = T / m

            s = h:h:10
            x = repeat(s, 1, length(s))
            y = Matrix(x')

            pars = [0.5, 1, 0.5, π/4, 2.5, 2.5]
            mats = make_matricies(n, α)

            r, ru, rv, rE = euler_vortex(x, y, 0, pars)
            pbar = Progress(m; showspeed=true)
            for i in 1:m
                r, ru, rv, rE = my_euler_rk4step(r, ru, rv, rE, h, k, mats)
                ProgressMeter.next!(pbar; showvalues=[(:total_iterations, m), (:current_iteration, i), (:time, (round(i*k, digits=5)))])
            end

            r0, ru0, rv0, rE0 = euler_vortex(x, y, T, pars)
            er = error(r, r0)
            eru = error(ru, ru0)
            erv = error(rv, rv0)
            erE = error(rE, rE0)
            err = maximum([er, eru, erv, erE])
            println("Maximum error was $(err)")

            push!(hs, log10(h))
            push!(errs, log10(err))
        end
        push!(out, [hs, errs])
    end
    return out
end


function vortex_convergence(αs, ns, data)
    """
    makes log-log convergence-plot of error for different filtering-parameters
    Uses vortex_convergence(αs, ns) run simulations
    """
    p = plot(title="Maximum error convergence", fontsize=64, size=(600, 300))
    for i in 1:length(αs)
        α = αs[i]
        hs, errs = data[i]
        hs = Float64.(hs)
        errs = Float64.(errs)

        X = zeros(length(hs), 2)
        X[:, 1] .= hs
        X[:, 2] .= 1
        slope, inter = X \ errs
        plot!(hs, errs, st=:scatter, color=:black, markershapes=:diamond, markersize=6, label=false)
        plot!(hs, inter .+ slope .* hs, linewidth=3, label=L"\alpha"*"=$(rpad(α, 5, "0")), Slope = $(round(slope, digits=3))")
    end
    plot!(xlabel=L"log_{10}(h)", ylabel=L"log_{10}(Error)", legend=:bottomright)
    savefig("figs/vortex_convergence.pdf")
    println("Saved convergence-plot")
end


function KelvinHelmholtz()
    """
    Simulates KelvinHelmholtz-instability with parameters specified in PS2.
    """
    println("Staring Kelving-Helmholtz")
    T = 1
    N = 256
    h = 1 / N
    α = 0.48

    m = Int(ceil(10T / 3h))
    k = T / m

    s = h:h:1
    x = repeat(s, 1, length(s))
    y = Matrix(x')

    mats = make_matricies(N, α)

    mask = @. abs(y - 0.5) < (0.15 + sin(2pi * x) / 200)
    r = ones(N, N)
    r[mask] .= 2
    u = r .- 1
    v = zeros(N, N)
    p = 3
    γ = 1.4
    E = @. p / ((γ - 1) * r) + (u^2 + v^2) / 2

    ru = r .* u
    rv = r .* v
    rE = r .* E

    pbar = Progress(m; showspeed=true)
    for i in 1:m
        r, ru, rv, rE = my_euler_rk4step(r, ru, rv, rE, h, k, mats)
        ProgressMeter.next!(pbar; showvalues=[(:total_iterations, m), (:current_iteration, i), (:time, (round(i*k, digits=5)))])
    end
    plot(x, y, r, st=:contourf, xlabel="x", ylabel="y", title="KelvinHelmholtz-instability, T=1")
    savefig("figs/KelvinHelmholtz.pdf")
    println("Saved KelvinHelmholtz-image")
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Part f
    αs = [0.499, 0.48]
    ns = [32, 64, 128]
    @time data = votrex_convergence(αs, ns)
    @time vortex_convergence(αs, ns, data)

    # part g
    @time KelvinHelmholtz()
end


