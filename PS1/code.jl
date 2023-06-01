using SparseArrays, Plots
using LaTeXStrings
Plots.pyplot()
println("Done importing")


# Problem 1:
function Poisson5(n, f, g)
    """
    A, b, x, y = assemblePoisson(n, f, g)

    Assemble linear system Au = b for Poisson's equation using finite differences.
    Grid size (n+1) x (n+1), right hand side function f(x,y), Dirichlet boundary
    conditions g(x,y).

    Taken from course page, UC Berkeley Math 228B
    credit to Per-Olof Persson <persson@berkeley.edu>
    """
    h = 1.0 / n
    N = (n+1)^2
    x = h * (0:n)
    y = x

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    A = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    b = zeros(N)

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if i == 1 || i == n+1 || j == 1 || j == n+1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, 1.0))
                b[row] = g(x[i],y[j])
            else
                # Interior nodes, 5-point stencil
                push!(A, (row, row, 4.0))
                push!(A, (row, umap[i+1,j], -1.0))
                push!(A, (row, umap[i-1,j], -1.0))
                push!(A, (row, umap[i,j+1], -1.0))
                push!(A, (row, umap[i,j-1], -1.0))
                b[row] = f(x[i], y[j])  * h^2
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A, b, x, y
end

function Poisson9(n, f, g)
    """
    A, b, x, y = Poisson9(n, f, g)

    Same as Possion5(), extended for 9-point stencil

    based on function 'assemblePoisson' from course page:
    UC Berkeley Math 228B,
    credit to Per-Olof Persson <persson@berkeley.edu>
    """
    h = 1.0 / n
    N = (n+1)^2
    x = h * (0:n)
    y = x

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    A = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    b = zeros(N)

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if i == 1 || i == n+1 || j == 1 || j == n+1
                # Dirichlet boundary condition, u = g
                push!(A, (row, row, 1.0))
                b[row] = g(x[i],y[j])
            else
                # Interior nodes, 9-point stencil
                push!(A, (row, row, 20.0))

                push!(A, (row, umap[i+1,j], -4.0))
                push!(A, (row, umap[i-1,j], -4.0))
                push!(A, (row, umap[i,j+1], -4.0))
                push!(A, (row, umap[i,j-1], -4.0))

                push!(A, (row, umap[i+1,j-1], -1.0))
                push!(A, (row, umap[i-1,j+1], -1.0))
                push!(A, (row, umap[i+1,j+1], -1.0))
                push!(A, (row, umap[i-1,j-1], -1.0))

                # 5-point stencil for nabla f_ij
                d5f = 4 * f(x[i], y[j])
                d5f -= f(x[i-1], y[j])
                d5f -= f(x[i+1], y[j])
                d5f -= f(x[i], y[j-1])
                d5f -= f(x[i], y[j+1])

                # rhs of linear equations
                b[row] = (f(x[i], y[j]) - d5f / 12) * h^2 * 6
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A, b, x, y
end

function assemblePoisson(n, f, g)
    """
    Just in case its really important the function is called assemblePoisson
    """
    Poisson9(n, f, g)
end

function testPoisson(n=40, Poisson=Poisson9)
    """
    error = testPoisson(n=20)

    Poisson test problem:
    - Prescribe exact solution uexact
    - set boundary conditions g = uexact and set RHS f = -Laplace(uexact)

    Solves and plots solution on a (n+1) x (n+1) grid.
    Returns error in max-norm.

    Based on function from course page,
    credit to Per-Olof Persson <persson@berkeley.edu>
    """
    uexact(x,y) = exp(-(4(x - 0.3)^2 + 9(y - 0.6)^2))
    f(x,y) = uexact(x,y) * (26 - (18y - 10.8)^2 - (8x - 2.4)^2)
    A, b, x, y = Poisson(n, f, uexact)

    # Solve + reshape solution into grid array
    u = reshape(A \ b, n+1, n+1)

    # Compute error in max-norm
    u0 = uexact.(x, y')
    error = maximum(abs.(u - u0))
    return error
end

function gridRefinement(n)
    """
    Finds error in max-norm for different grid-sizes ns
    Plots log(error) as function of step-length
    Performs regression to determine power law
    Does this for both 5-point stencil and 9-point stencil

    Saves plot
    """
    println("Performing gridRefinement test")
    h = log10.(1 ./ n)
    errs5 = zeros(length(n))
    errs9 = zeros(length(n))
    for i = 1:length(n)
        errs9[i] = log10(testPoisson(n[i]))
        errs5[i] = log10(testPoisson(n[i], Poisson5))
    end

    # determine slope of decreasing error using linear regression
    exclude = 2  # points to exclude from slope estimation
    X = zeros(length(n) - exclude, 2)
    X[:, 1] .= h[exclude+1:end]
    X[:, 2] .= 1.
    slope9, i9 = (X \ errs9[exclude+1:end])
    slope5, i5 = (X \ errs5[exclude+1:end])

    println("Plotting result")
    plot(h, errs9, linewidth=2, markershapes=:cross, markersize=8, label="9-point. Slope = $(round(slope9, digits=3))")
    plot!(h, errs5, linewidth=2, markershapes=:cross, markersize=8, label="5-point. Slope = $(round(slope5, digits=3))")
    plot!(title="Error as function of step-size h", xlabel=L"\log_{10}(h)", ylabel=L"\log_{10}(Error)", legend=:bottomright)
    savefig("figs/gridRefinement.pdf")
end

# Problem 2:
function buildA(L, B, H, n)
    """
    A, b, x, y = buildA

    Assembles Au=b for Poisson equation in computational domin xi and eta
    calculates physical grid x and y
    """
    # computational domain tools
    h = 1.0 / n
    N = (n+1)^2
    xi = h * (0:n)
    eta = xi

    # constants from problem set
    D = (L - B) / 2
    A = sqrt(D^2 - H^2)

    gamma(i, j) = B/2 + A * eta[j] # useful definition
    # Domain transformations
    x(i, j) = xi[i] * gamma(i, j)
    y(i, j) = H * eta[j]

    # horrible expressions
    Jac(i, j) = H * gamma(i, j)
    a(i, j) = A^2 * xi[i]^2 + H^2
    b(i, j) = A * xi[i] * gamma(i, j)
    c(i, j) = gamma(i, j)^2
    d(i, j) = 0
    e(i, j) = 2 * A^2 * xi[i]

    # creat meshes for x and y
    X = [(x(i, j), y(i, j)) for i=1:n+1, j=1:n+1]
    xvec = map(x->x[1], X)
    yvec = map(x->x[2], X)

    umap = reshape(1:N, n+1, n+1)     # Index mapping from 2D grid to vector
    Amat = Tuple{Int64,Int64,Float64}[]  # Array of matrix elements (row,col,value)
    bvec = zeros(N)

    # Main loop, insert stencil in matrix for each node point
    for j = 1:n+1
        for i = 1:n+1
            row = umap[i,j]
            if j == 1 || i == n+1
                # Dirichlet boundary condition, u = 0 at P1P2, P2P3
                push!(Amat, (row, row, 1.0))
                bvec[row] = 0

            elseif i == 1
                # von Neumann condition, u'=0 at P1P4
                # normal vector same in both physical and computational domain
                push!(Amat, (row, umap[i,j], -1.5))
                push!(Amat, (row, umap[i+1,j], 2.0))
                push!(Amat, (row, umap[i+2,j], -0.5))
                bvec[row] = 0

            elseif j == n+1
                # von Neumann condition, u'=0 at P3P4
                # normal derivative components
                nxi = 1 / 2 * A * xi[i] / Jac(i, j)
                neta = 1 / H

                # du/deta. 3-point left boundary scheme
                push!(Amat, (row, umap[i,j], -1.5 * neta))
                push!(Amat, (row, umap[i,j-1], 2.0 * neta))
                push!(Amat, (row, umap[i,j-2], -0.5 * neta))

                # du/dxi. 2-point central derivative
                push!(Amat, (row, umap[i+1,j], 1 * nxi))
                push!(Amat, (row, umap[i-1,j], -1 * nxi))
                bvec[row] = 0

            else
                # Interior nodes
                push!(Amat, (row, row, -2 * (a(i, j) + c(i, j))  ) )

                push!(Amat, (row, umap[i+1,j], a(i, j) + e(i, j) * h / 2))
                push!(Amat, (row, umap[i-1,j], a(i, j) - e(i, j) * h / 2))
                push!(Amat, (row, umap[i,j+1], c(i, j)))
                push!(Amat, (row, umap[i,j-1], c(i, j)))

                push!(Amat, (row, umap[i+1,j+1], -b(i, j) / 2))
                push!(Amat, (row, umap[i-1,j-1], -b(i, j) / 2))
                push!(Amat, (row, umap[i+1,j-1], b(i, j) / 2))
                push!(Amat, (row, umap[i-1,j+1], b(i, j) / 2))

                # rhs of linear equations
                bvec[row] = -Jac(i, j)^2 * h^2
            end
        end
    end

    # Create CSC sparse matrix from matrix elements
    Amat = sparse((x->x[1]).(Amat), (x->x[2]).(Amat), (x->x[3]).(Amat), N, N)

    return Amat, bvec, xvec, yvec
end

function channelflow(L, B, H, n)
    """
    Q, x, y, u = channelflow(L, B, H, n)

    Gets A and b from buildA to solve for u
    Calculates Q using 2D-trapezoidal scheme
    """
    A, b, x, y = buildA(L, B, H, n)
    u = reshape(A \ b, n+1, n+1)

    Q = u[1, 1] + u[1, end] + u[end, 1] + u[end, end]
    Q += 2 * (sum(u[1, :]) + sum(u[end, :]) + sum(u[:, 1]) + sum(u[:, end]))
    Q += 4 * sum(u[2:end-1, 2:end-1])
    Q /= 4 * n^2
    return Q, x, y, u
end

function plotflow(x, y, u, text)
    """
    Plots line-contour of u with xy-grid, as well as filled contour of u in same figure
    """
    println("Plotting contours")
    p = plot(title=text, xlabel="x", ylabel="y", layout=2, size=(1000, 600), fontsize=14)
    plot!(p[1], x, y, u, c=:viridis, st=:contour, fill=false, levels=12, contour_labels=true)
    plot!(p[1], x, y, label=false, color="black", lw=1, alpha=0.2)
    plot!(p[1], x', y', label=false, color="black", lw=1, alpha=0.2)
    plot!(p[2], x, y, u, c=:viridis, st=:contour, fill=true, levels=201)
    savefig("figs/contour.pdf")
end

function testChannelflow(n=20)
    """ Makes a nice contour plot so show my implementation is correctish """
    L = 3.
    B = 0.5
    H = 1.

    Q, x, y, u = channelflow(L, B, H, n)
    plotflow(x, y, u, "Channelflow")
end

function channelConvergence(Bs, start_n, n_cnt, excl)
    """
    Determines convergence rate for Q for different B and makes convergence plot
    Arguments:
        B: array of B-values to loop over. Makes a plot for each of these
        start_n: First n-value. Goes up by a factor of 2
        n_cnt: how many n-values to calculate Q for.
        excl: how many datapoints to exclude when determining slope

    Starts at n=start_n, then n goes up with a factor 2 n_cnt times.
    The true Q is calculated from the next n-value (twice as large as the last one)
    Then use linear regression to determine slope of points in log-log plot
    Plots the results in 1 plot for each B-value
    """
    println("Start channelConvergence")
    L = 3. # unchanged constants
    H = 1.
    reference = start_n * 2^n_cnt
    plots = []  # container for each B-plot
    for B in Bs
        trueQ, _, _, _ = channelflow(L, B, H, reference)
        println("B = $(B), Q = $(trueQ)")
        hs = zeros(n_cnt)
        errs = zeros(n_cnt)
        n = start_n
        for i in 1:n_cnt
            Q, _, _, _ = channelflow(L, B, H, n)
            errs[i] = log10(abs(Q - trueQ) / trueQ)
            hs[i] = log10(1 / n)
            n *= 2
        end
        X = zeros(length(hs) - excl, 2)
        X[:, 1] .= hs[excl+1:end]
        X[:, 2] .= 1
        slope, inter = (X \ errs[excl+1:end])
        p = plot(hs, errs, title="Relative error in "*L"\hat{Q}"*" for B = $(B). Q = $(round(trueQ, digits=4))", st=:scatter, markershapes=:cross, markersize=8, label="")
        plot!(hs, inter .+ slope .* hs, linewidth=2, label="Slope = $(round(slope, digits=3))")
        push!(plots, p)
    end
    l = @layout [a; b; c]
    plot(plots..., layout=l, size=(600, 800), xlabel=L"\log_{10}(h)", ylabel=L"\log_{10}(Error)", legend=:bottomright)
    savefig("figs/convergence.pdf")
end

if abspath(PROGRAM_FILE) == @__FILE__
    gridRefinement(collect([2; 10:10:200]))     # Problem 1b
    testChannelflow()                           # Problem 2c
    channelConvergence([0, 0.5, 1], 10, 6, 1)   # problem 2d
end
