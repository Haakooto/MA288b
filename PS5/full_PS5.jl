using PyCall
using Plots, TriplotRecipes
using SparseArrays, LinearAlgebra
using LaTeXStrings

"""
Code for Problem Set 5 Math228b

First use finite element method with linear elements to solve Helmholtz problem -u''(x,y)=k^2u
Makes a convergence test of method on square, then solves on more comple domain

Then use quadratic elements to solve Poisson problem -u''(x,y)=1
Makes a convergence test on a square domain

Code is structured with linear FEM-code first, with the plot-generating function first,
followed by the actual method-implementating functions. Then the same for the quadratic FEM.
The meshgenerator developed over the last few problem sets is at the bottom of the file, unchanged since PS4

tplot() has been changed to work with Plots.jl, instead of PyPlot.jl. TriplotRecipes.jl is required
"""

################
# Part 2 and 3 #
################

u_exact(x, k) = exp(-im * k * x)  # exact solution to from 2a)
cdot(x, y) = real.(dot(conj(x)', y))  # Born-rule
H(u, B) = cdot(u, B * u)  # Intensity

"""
    convergence_helmholtz()

For different degrees of meshrefinement, find solution to Helmholtz on a square
Use exact solution to make convergence plot
"""
function convergence_helmholtz()
    println("Starting Helmholtz convergence test")
    xmax = 5
    k = 6
    hmax = 0.3
    nrefmax = 4

    errs = zeros(nrefmax)
    hs = zeros(nrefmax)

    for nref = 1:nrefmax
        pv = [0 0; xmax 0; xmax 1; 0 1; 0 0]
        p, t, e = pmesh(pv, hmax, nref)

        u, _ = solve_helmholtz(p, t, k)
        exact = u_exact.(p[:, 1], k)
\
        errs[nref] = log2(maximum(abs.(exact - u)))
        hs[nref] = log2(hmax / (2^nref))
    end

    X = zeros(length(hs)-1, 2)
    X[:, 1] .= hs[2:end]
    X[:, 2] .= 1
    slope, inter = X \ errs[2:end]

    plot(hs, errs, c="black", markershapes=:diamond, st=:scatter, markersize=8, label=false)
    plot!(hs, inter .+ slope .* hs, c="red", lw=4, label="Slope = $(round(slope, digits=3))")
    plot!(title="Convergence of method", xaxis="log(h)", yaxis="log(error)", xflip=true, legend=:bottomleft)
    savefig("figs/convergence_helmholtz.pdf")
    println("saved figure figs/convergence_helmholtz.pdf")
    println()
end

"""
    frequency_response()

Defines a double slitted domain, and solves helmholtz problem
for many different wavenumbers k.
Makes a plot of the mesh itself,
and a figure with the intensity H at the output boundary as function of k,
as well as a figure of the solution u in the domain for the value of k
giving highest and lowest intensity H
"""
function frequency_response()
    println("Starting waveguide frequency response")
    # fully parametrerized rectangular domain with 2 slits
    xmax = 5   # length of domain
    ymax = 1   # height of domain
    sw = 0.2   # slit width
    sh = 0.2   # slit height
    s1 = 3     # centre of slit 2
    s2 = 2     # centre of slit 1
    pv = [0 0; xmax 0; xmax ymax; s1+sw/2 ymax; s1+sw/2 sh; s1-sw/2 sh; s1-sw/2 ymax; s2+sw/2 ymax; s2+sw/2 sh; s2-sw/2 sh; s2-sw/2 ymax; 0 ymax; 0 0]
    p, t, e = pmesh(pv, 0.2, 2)

    tplot(p, t; title="Mesh", xlabel="x", ylabel="y")
    savefig("figs/mesh.pdf")
    println("saved figure figs/mesh.pdf")

    ks = []
    Is = []

    Matrices = get_helmholtz_matrices(p, t)
    for k = 6:0.01:6.5
        u, intensity = solve_helmholtz(Matrices, k)

        push!(ks, k)
        push!(Is, intensity)
    end
    Hk = plot(ks, log.(Is), xaxis="wavenumber k", yaxis="Intensity log(H)", lw=5, label=:none, title="Intensity vs wavenumber k")

    l = @layout [a{0.4h}; b; c]

    _, mx = findmax(Is)
    _, mn = findmin(Is)
    umx, Hmx = solve_helmholtz(Matrices, ks[mx])
    umn, Hmn = solve_helmholtz(Matrices, ks[mn])

    cmx = maximum(abs.(real.(umx)))
    cmn = maximum(abs.(real.(umn)))

    Pmx = tplot(p, t; u=real.(umx), c=:berlin, clim=(-cmx, cmx), xaxis="x", yaxis="y", title="Real part of solution with higest intensity, k = $(ks[mx]), H = $(round(Hmx, digits=3)).")
    Pmn = tplot(p, t; u=real.(umn), c=:berlin, clim=(-cmn, cmn), xaxis="x", yaxis="y", title="Real part of solution with lowest intensity, k = $(ks[mn]), H = $(round(Hmn, digits=3)).")
    P = plot(Hk, Pmx, Pmn, layout=l, size=(650, 800))
    savefig("figs/IvK_mx_mn.pdf")
    println("saved figure figs/IvK_mx_mn.pdf")
    println()
end

"""
    u, H = solve_helmholtz(p, t, k)

Solves helmholtz problem on mesh p with triangulation k and wavenumber k
returns full solution u in domain, and intensity H at output bounadry
"""
function solve_helmholtz(p, t, k)
    ein, eout, ewall = waveguide_edges(p, t)
    K, M, Bin, Bout, bin = femhelmholtz(p, t, ein, eout)

    A = K - k^2 * M + im * k * (Bin + Bout)
    b = 2im * k * bin

    u = A \ b
    return u, H(u, Bout)
end

"""
    K, M, Bin, Bout, bin = get_helmholtz_matrices(p, t)

returns just the matrices for solving helmholtz
"""
function get_helmholtz_matrices(p, t)
    ein, eout, ewall = waveguide_edges(p, t)
    femhelmholtz(p, t, ein, eout)
end

"""
    u, H = solve_helmholtz(Mats, k)

Using precomputed matrices, compute solution to Helmholtz problem for wavenumber k
"""
function solve_helmholtz(Mats, k)
    K, M, Bin, Bout, bin = Mats

    A = K - k^2 * M + im * k * (Bin + Bout)
    b = 2im * k * bin

    u = A \ b
    return u, H(u, Bout)
end

"""
    ein, eout, ewall = waveguide_edges(p, t, xmax)

finds different component of boundary for mesh p with triangulation t
ein is vertical edges at x=0
eout is vertical edges at x=xmax
ewall is remaining boundary edges
"""
function waveguide_edges(p, t, xmax=5)
    edges, bnidx, _ = all_edges(t)
    ein, eout, ewall = [], [], []
    tol = 1e-8  # just for extra precaution, should be unnessisary
    for k = bnidx  # only loop over actual boundary nodes
        dy = abs(p[edges[k, 2], 2] - p[edges[k, 1], 2])
        if dy > tol  # check verticality
            x = p[edges[k, 1], 1]
            if x < tol
                push!(ein, k)
            elseif x > xmax - tol
                push!(eout, k)
            end
        else
            push!(ewall, k)
        end
    end
    return edges[ein, :], edges[eout, :], edges[ewall, :]
end

"""
    K, M, Bin, Bout, bin = femhelmholtz(p, t, ein, eout)

Returns matries for Helmholtz problem on points p with triangulation t,
With homogeneous dirichlet boundary on eout,
in-homogeneous dirichlet boundary on ein
and natural boundary conditons on the remaining bounadry
"""
function femhelmholtz(p, t, ein, eout)
    n = size(p, 1)
    K, M, Bin, Bout = [], [], [], []
    bin = zeros(n)

    # integrate in the interiour
    for k = 1:size(t, 1)
        sK, sM = stamp_H(p, t[k, :])
        for i = 1:3, j = 1:3
            push!(K, (t[k, i], t[k, j], sK[i, j]))
            push!(M, (t[k, i], t[k, j], sM[i, j]))
        end
    end

    # integrate along input boundary
    _sB() = [2 1; 1 2] / 6
    for k in 1:size(ein, 1)
        sB = _sB()
        d = edge_stamp!(sB, p, ein[k, :])
        for i = 1:2, j = 1:2
            push!(Bin, (ein[k, i], ein[k, j], sB[i, j]))
        end
        bin[ein[k, 1]] += d / 2
        bin[ein[k, 2]] += d / 2
    end

    # integrate along output boundary
    for k in 1:size(eout, 1)
        sB = _sB()
        edge_stamp!(sB, p, eout[k, :])
        for i = 1:2, j = 1:2
            push!(Bout, (eout[k, i], eout[k, j], sB[i, j]))
        end
    end

    sparsify(A, n=n) = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), n, n)
    return sparsify(K), sparsify(M), sparsify(Bin), sparsify(Bout), bin
end

"""
    sK, sM = stamp_H(p, t)

Returns stamp matrices for K and M for triangle p[t]
"""
function stamp_H(p, t)
    x1, y1 = p[t[1], :]
    x2, y2 = p[t[2], :]
    x3, y3 = p[t[3], :]

    area = (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1) / 2
    C = inv([1 x1 y1; 1 x2 y2; 1 x3 y3])

    sK = zeros(3, 3)
    for i = 1:3
        sK[i, i] = C[2, i]^2 + C[3, i]^2  # diagonal
        for j = 1:i-1  # off-diagonal
            sK[i, j] = C[2, i] * C[2, j] + C[3, i] * C[3, j]
            sK[j, i] = sK[i, j]  # symmetry
        end
    end
    sK *= area
    sM = (ones(3) .+ I(3)) / 12 * area

    return sK, sM
end

"""
    d = edge_stamp!(B, p, t)

calculates distance d between points p[t]
multiplies B with d in-place
return d
"""
function edge_stamp!(B, p, t)
    d = norm(p[t[1], :] - p[t[2], :])
    B .*= d
    return d
end


##########
# Part 4 #
##########

"""
    p, t, e = p2mesh(p, t)

Makes mesh for quadratic fempoi solver
Adds midpoints to edges of mesh, and adds these to the respective triangles
"""
function p2mesh(p, t)
    # copy t2 such that it can hold 6 points pr element
    t2 = zeros(Int64, size(t, 1), 6)
    t2[:, 1:3] = t
    p2 = copy(p)
    e2 = boundary_nodes(t)

    counter = ones(Int64, size(t, 1)) * 3  # counts found points pr element
    num_points = size(p, 1)  # number of points in p

    edges, bnidx, emap = all_edges(t)
    for edge in emap  # for every edge
        if edge != 0  # if edge has not already been done
            # Find midpoint and add to p2
            q1 = p[edges[edge, 1], :]
            q2 = p[edges[edge, 2], :]
            new = q1 + 0.5(q2 - q1)
            p2 = [p2; new']
            num_points += 1

            # find every occurance of edge in emap
            in_triangles = findall(x -> x == edge, emap)
            if length(in_triangles) == 1
                e2 = [e2; num_points]
            end
            for triangle in in_triangles  # loop over occurances
                emap[triangle] = 0  # mark edge as done
                T = triangle[1]  # find corresponding triangle
                counter[T] += 1  # increment triangle counter
                t2[T, counter[T]] = num_points  # insert last point (new) into triangulation
            end
        end
    end
    return p2, t2, e2
end

"""
    convergence_fempoi2()

Makes a convergence plot for fempoi2
"""
function convergence_fempoi2()
    println("Starting fempoi2 convergence test")
    hmax = 0.3
    nrefmax = 4
    pv = [0 0; 1 0; 1 1; 0 1; 0 0]
    num_pts = size(pmesh(pv, hmax, 0)[1], 1)

    exact_sol = fempoi2(p2mesh(pmesh(pv, hmax, nrefmax)[1:2]...)...)

    errors = zeros(nrefmax)
    hs = zeros(nrefmax)
    for nref = nrefmax-1:-1:0
        p, t, e = pmesh(pv, hmax, nref)
        p2, t2, e2 = p2mesh(p, t)
        sol = fempoi2(p2, t2, e2)

        diff = sol[1:num_pts] - exact_sol[1:num_pts]
        errors[nref+1] = maximum(abs.(diff))
        hs[nref+1] = hmax / (2^nref)
    end
    hs = log10.(hs)
    errors = log10.(errors)
    X = zeros(length(hs), 2)
    X[:, 1] .= hs
    X[:, 2] .= 1
    slope, inter = X \ errors
    plot(hs, errors, c="black", markershapes=:diamond, st=:scatter, markersize=8, label=false)
    plot!(hs, inter .+ slope .* hs, c="red", lw=4, label="Slope = $(round(slope, digits=3))")
    P = plot!(title="Convergence of fempoi2", xaxis=L"log_{10}(h)", yaxis=L"log_{10}(error)", legend=:bottomright)
    savefig("figs/fempoi2_convergence.pdf")
    println("saved figure figs/fempoi2_convergence.pdf")
    println()
end

"""
    u = fempoi2(p, t, e)

Solves u''(x, y) = 1 on points p with elements t,
with homogeneous Dirichlet conditions on e
"""
function fempoi2(p, t, e)
    n = size(p, 1)
    A = []
    b = zeros(n)
    for k = 1:size(t, 1)
        sA, sb = get_stamp(p, t[k, :])
        for i = 1:6, j = 1:6
            push!(A, (t[k, i], t[k, j], sA[i, j]))
        end
        b[t[k, :]] += sb
    end
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), n, n)

    # Imposing homogeneous Dirichlet conditions
    for edge in e
        A[edge, :] .= 0
        A[:, edge] .= 0
        A[edge, edge] = 1
        b[edge] = 0
    end
    A = sparse(A) * 1.0
    A \ b
end

"""
    sA, sb = get_stamp(p, tk)

Returns stamps for A and b for element Tk
for solving u''(x, y) = 1
"""
function get_stamp(p, t)
    _area = area(p[t[1:3], :]) / 2

    V = zeros(6, 6)
    V[:, 1] .= 1
    V[:, 2] = p[t, 1]
    V[:, 3] = p[t, 2]
    V[:, 4] = p[t, 1] .^ 2
    V[:, 5] = p[t, 2] .^ 2
    V[:, 6] = p[t, 1] .* p[t, 2]
    C = inv(V)

    A = zeros(6, 6)
    b = zeros(6)
    for i = 1:6
        A[i, i] = quad_A(p[t[1:3], :], C[:, i], C[:, i])
        b[i] = quad_b(p[t[1:3], :], C[:, i])
        for j = 1:i-1
            A[i, j] = quad_A(p[t[1:3], :], C[:, i], C[:, j])
            A[j, i] = A[i, j]
        end
    end
    (A, b) .* _area
end

delta(i, j) = Int64(i == j) / 2  # half of delta function
xi(X, i) = sum([X[j] / 6 + delta(i, j) * X[j] for j = 1:3])

"""
    S = quad_A(X, Ci, Cj)

Integrates f'(x, y)g'(x, y) over a triangle of area 1, where
f and g are bivariate quadratic polynomials with coefficients Ci and Cj
"""
function quad_A(X, Ci, Cj)
    s = 0
    for i = 1:3
        x = xi(X[:, 1], i)
        y = xi(X[:, 2], i)
        s += (Ci[2] + 2Ci[4]*x + Ci[6]*y) * (Cj[2] + 2Cj[4]*x + Cj[6]*y) / 3
        s += (Ci[3] + 2Ci[5]*y + Ci[6]*x) * (Cj[3] + 2Cj[5]*y + Cj[6]*x) / 3
    end
    return s
end

"""
    S = quad_b(X, Ci)

Integrates (a+bx+cy+dx^2+ey^2+fxy) over a triangle of area 1
using a numerical quadrature scheme
"""
function quad_b(X, Ci)
    a, b, c, d, e, f = Ci
    s = 0
    for i = 1:3
        x = xi(X[:, 1], i)
        y = xi(X[:, 2], i)
        s += a + b*x + c*y + d*x^2 + e*y^2 + f*x*y
    end
    return s
end


"""
    tplot(p, t, u=nothing, pts=false)

If `u` == nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` == solution vector: Plot filled contour color plot of solution `u`.
If 'pts', scatter verticies on grid.
Extra kwargs are passed to plotting funcs

Changed to use Plots.jl instead of PyPlot.
TriplotRecipes.jl is required
"""
function tplot(p, t; u=nothing, pts=false, kwargs...)
    if size(t, 2) == 6
        t = t[:, 1:3]
    end
    if u === nothing
        fig = trimesh(p[:, 1], p[:, 2], t'; aspect_ratio=:equal, kwargs...)
    else
        fig = tripcolor(p[:, 1], p[:, 2], u, t'; aspect_ratio=:equal, kwargs...)
    end
    if pts
        plot!(p[:, 1], p[:, 2], st=:scatter, markersize=8, labels=false)
    end
    return fig
end


#################
# Meshgenerator #
#################

# All code below is copied from previous PS, with little to no changes


"""
    p = add_edge_points(pv, hmax)

Adds points along the edge of the polygon with verticies pv,
such that distance between the points are less than hmax
"""
function add_edge_points(pv, hmax)
    for j = 2:size(pv)[1]  # assume pv is ordered
        direction = pv[j, :] - pv[j-1, :]
        dist = norm(direction)
        n = dist ÷ hmax
        h = 1 / (n + 1)
        for i = 1:n
            new = @. pv[j-1, :] + direction * i * h
            pv = [pv; new']
        end
    end
    return pv[2:end, :]  # first polygon is closed, so remove dupliated point
end


"""
    T = out_triangles(p, T, e)

returns the list of triangles without the ones outside the polygon
Calculates a single point in each triangle, and uses inpolygon to check if theyre outside
    p: points in mesh
    T: indicies in p forming each triangle
    e: points on edge of polygon
"""
function out_triangles(p, T, e)
    test_points = zeros(size(T)[1], 2)
    for t = 1:size(T)[1]
        a, b, c = T[t, :]
        test_points[t, :] = p[a, :] + 0.5(p[b, :] - p[a, :]) + 0.25(p[c, :] - p[b, :])
    end
    # e is a closed polygon, so check for all but first point on edge
    inside = inpolygon(test_points, e[2:end, :])
    return T[inside, :]
end

"""
    t = triangulate(p, t)

Full triangulation cycle:
1) call delaunay to triangulate
2) remove triangles outside edge
3) remove degenerate triangles
"""
function triangulate(p, e)
    t = delaunay(p)
    t = out_triangles(p, t, e)
    dedegenerate(p, t)
end

"""
    t = dedegenerate(p, t)

removes degenerate triangles from triangulation t
a degenerate triangle has area less than 1e-12
"""
function dedegenerate(p, t)
    bad = []
    for m in 1:size(t, 1)
        if area(p[t[m, :], :]) < 1e-12
            push!(bad, m)
        end
    end
    for k in reverse(bad)
        t = t[1:end.!=k, :]
    end
    return t
end

"""
    A = area(p[T[i, :], :])

    Calculates area of triangle with vertices X
"""
function area(X)
    x1, x2, x3, y1, y2, y3 = reshape(X, 6)
    x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1
end

"""
    idx, t = largest_area(p, t, hmax)

For every triangle, calc area, if larger than largest so far, save index
index of largest area is retruned.
degenerate triangles are removed.
"""
function largest_area(p, T, hmax)
    worst = hmax^2
    idx = false
    for m = 1:size(T, 1)
        A = area(p[T[m, :], :])
        if A > worst
            worst = A
            idx = m
        end
    end
    return idx
end

"""
p_mid = circumcentre(X...)

returns the circumcentre for a triangle of 3 points
"""
function circumcentre(ax, bx, cx, ay, by, cy)
    D = 2(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    A = ax^2 + ay^2
    B = bx^2 + by^2
    C = cx^2 + cy^2

    Ux = (A * (by - cy) + B * (cy - ay) + C * (ay - by)) / D
    Uy = (A * (bx - cx) + B * (cx - ax) + C * (ax - bx)) / D

    return [Ux -Uy]  # defined Uy negatively
end

"""
p = refine(p, t)

Adds a point in the middle of each edge.
"""
function refine(p, t)
    edges, _, _ = all_edges(t)
    for e = 1:size(edges)[1]
        p1 = p[edges[e, 1], :]
        p2 = p[edges[e, 2], :]
        new = p1 + 0.5(p2 - p1)
        p = [p; new']
    end
    return p
end

"""
    p, t, e = pmesh(pv, hmax, nref)

Makes a mesh of polygon pv, such that no edge length is larger than hmax
Then refines mesh by inserting midpoint between points nref times
p: (N, 2) = all points in refined mesh
t: (T, 3) = triangles, p[t[k, :], :] are corners of triangle k
e: (Ne, 1) = all points that is on boundary of polygon pv
"""
function pmesh(pv, hmax, nref)
    p = add_edge_points(pv, hmax)
    too_large = true
    while too_large
        t = triangulate(p, pv)
        idx = largest_area(p, t, hmax)

        if idx != false
            p = [p; circumcentre(p[t[idx, :], :]...)]
        else
            too_large = false
        end
    end

    t = triangulate(p, pv)
    for _ in 1:nref
        p = refine(p, t)
        t = triangulate(p, pv)
    end

    e = boundary_nodes(t)
    return p, t, e
end

"""
    t = delaunay(p)

Delaunay triangulation `t` of N x 2 node array `p`.
"""
function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri[:Triangulation](p[:, 1], p[:, 2])
    return Int64.(t[:triangles] .+ 1)
end

"""
inside = inpolygon(p, pv)

Determine if each point in the N x 2 node array `p` is inside the polygon
described by the NE x 2 node array `pv`.
"""
function inpolygon(p, pv)
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    inside = [poly[:contains_point](p[ip, :]) for ip = 1:size(p, 1)]
end

"""
    edges, boundary_indices, emap = all_edges(t)

Find all unique edges in the triangulation `t` (ne x 2 array)
Second output is indices to the boundary edges.
Third output emap (nt x 3 array) is a mapping from local triangle edges
to the global edge list, i.e., emap[it,k] is the global edge number
for local edge k (1,2,3) in triangle it.
"""
function all_edges(t)
    etag = vcat(t[:, [1, 2]], t[:, [2, 3]], t[:, [3, 1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t, 1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end, 1:2] - etag[1:end-1, 1:2] .== 0, dims=2)[:]
    keep = .![false; dup]
    edges = etag[keep, 1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:, 3])
    emap = reshape(emap, :, 3)
    dup = [dup; false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end

"""
    e = boundary_nodes(t)

Find all boundary nodes in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges(t)
    return unique(edges[boundary_indices, :][:])
end


###############################################
# Running code to produce figures in write-up #
###############################################
convergence_helmholtz()  # 2d
frequency_response()     # 3b and 3c
convergence_fempoi2()    # 4c
