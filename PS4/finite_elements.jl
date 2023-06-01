using PyPlot, PyCall
using SparseArrays, LinearAlgebra
using LaTeXStrings

"""
Code for PS4
Structure:
    Code for part 1, 2, and 3 placed sequentialy down.
    At very end I have placed the meshgenerator from PS3, but with a bug-fix
"""

""" Part 1 """

# Hermite splines
H0(x) = 2x^3 - 3x^2 + 1
H1(x) = 3x^2 - 2x^3
H0t(x) = x^3 - 2x^2 + x
H1t(x) = x^3 - x^2

# basis functions
function phi_1(x)
    if x < 1 / 2
        return H1(2x)
    else
        return H0(2x - 1)
    end
end
function phi_2(x)
    if x < 1 / 2
        return H1t(2x)
    else
        return H0t(2x - 1)
    end
end

u_exact(x) = 4x^5 - 5x^4 - 2x^3 + 3x^2
u_num(x, u1, u2) = u1 * phi_1(x) + u2 * phi_2(x)

function part_1()
    n = 101
    x = (0:n) / n

    A = [12 0; 0 4]
    b = [-60, -38]
    u1, u2 = A \ b

    plot(x, u_num.(x, u1, u2), "k", lw=3, label=L"u_{num}")
    plot(x, u_exact.(x), "r--", lw=3, label=L"u_{exact}")
    xlabel("x")
    ylabel("y")
    title("FE numerical solution, and exact for BVP (1)-(2)")
    legend()
    savefig("figs/uplot.pdf")
end

""" Part 2 """

"""
    stampA, stampb = get_stamp(p, t)

Assemble local stamp for A and b, from points p[t]
Calculate area of triangle from verticies
Calculate coefficient matrix by inverting vandermonde

"""
function get_stamp(p, t)
    x1, y1 = p[t[1], :]
    x2, y2 = p[t[2], :]
    x3, y3 = p[t[3], :]

    area = x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1
    C = inv([1 x1 y1; 1 x2 y2; 1 x3 y3])

    # Stamp
    A = zeros(3, 3)
    for i = 1:3
        A[i, i] = C[2, i]^2 + C[3, i]^2  # diagonal
        for j = 1:i-1  # off-diagonal
            A[i, j] = C[2, i] * C[2, j] + C[3, i] * C[3, j]
            A[j, i] = A[i, j]  # symmetry
        end
    end
    A *= area
    b = ones(3) * area / 3

    return A, b
end

"""
    u = fempoi(p, t, e)

Solve poissons equation on polygon p with elements t
homogeneous Dirichlet conditions are imposed on e: u(e) = 0
homogeneous Neumann conditions are assumed on other boundaries.

"""
function fempoi(p, t, e)
    n = size(p, 1)
    A = []
    b = zeros(n)

    # Stamping method
    for k = 1:size(t, 1)
        sA, sb = get_stamp(p, t[k, :])

        for i = 1:3, j = 1:3
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
    # Solve PDE
    A = sparse(A) * 1.0
    A \ b
end

function test_poisson()
    # Square, Dirichlet left/bottom
    pv = Float64[0 0; 1 0; 1 1; 0 1; 0 0]
    p, t, e = pmesh(pv, 0.15, 0)
    e = e[@. (p[e, 1] < 1e-6) | (p[e, 2] < 1e-6)]
    u = fempoi(p, t, e)
    tplot(p, t, u=u, levels=10)
    savefig("figs/square.pdf")

    # Circle, all Dirichlet
    n = 32
    phi = 2pi * (0:n) / n
    pv = [cos.(phi) sin.(phi)]
    p, t, e = pmesh(pv, 2pi / n, 0)
    u = fempoi(p, t, e)
    tplot(p, t, u=u, levels=10)
    savefig("figs/circle.pdf")

    # Generic polygon geometry, mixed Dirichlet/Neumann
    x = 0:0.1:1
    y = 0.1 * (-1) .^ (0:10)
    pv = [x y; 0.5 0.6; 0 0.1]
    p, t, e = pmesh(pv, 0.04, 0)
    e = e[@. p[e, 2] > (0.6 - abs(p[e, 1] - 0.5) - 1e-6)]
    u = fempoi(p, t, e)
    tplot(p, t, u=u, levels=10)
    savefig("figs/generic.pdf")

    # Pac-man polygon, Dirichlet on corners, Neumann elsewhere
    pv = Float64[0 0; 1 0; 0.5 0.5; 1 1; 0 1; 0 0]
    p, t, e = pmesh(pv, 0.15, 0)
    u = fempoi(p, t, 1:5)
    tplot(p, t, u=u, levels=10)
    savefig("figs/nepal.pdf")
end

""" Part 3 """

"""
errors = poiconv(pv, hmax, nrefmax)

Calculates difference in solutions of fempoi()
    nref= 0,...,nrefmax-1
    nref = nrefmax
for a triangulated polygon pv, with initial edge-length hmax and refined nref times
"""
function poiconv(pv, hmax, nrefmax)
    errors = zeros(nrefmax)
    exact_sol = fempoi(pmesh(pv, hmax, nrefmax)...)
    for nref = nrefmax-1:-1:0
        p, t, e = pmesh(pv, hmax, nref)
        sol = fempoi(p, t, e)
        n = size(p)[1]
        diff = sol - exact_sol[1:n, :]
        errors[nref+1] = maximum(abs.(diff))
    end

    return errors
end


function part_3()
    hmax = 0.15
    nrefmax = 5
    pv_square = Float64[0 0; 1 0; 1 1; 0 1; 0 0]
    pv_polygon = Float64[0 0; 1 0; 0.5 0.5; 1 1; 0 1; 0 0]

    errors_square = poiconv(pv_square, hmax, nrefmax)
    errors_polygon = poiconv(pv_polygon, hmax, nrefmax)
    errors = [errors_square errors_polygon]

    clf()
    rates = @. log2(errors[end-1, :]) - log2(errors[end, :])
    l = "slope = "
    label = ["Square.  " * l, "Polygon. " * l] .* string.(round.(rates, digits=5))
    loglog(hmax ./ (2 .^ collect(0:nrefmax-1)), errors, "o-", label=label, lw=3, ms=10)
    legend()
    xlabel(L"h \ \ \ [h_{max}]")
    ylabel("log2(error)")
    title("Convergence plot for nrefmax = $(nrefmax)")
    savefig("figs/convplot.pdf")
end


""" Meshgenerator """

"""
    tplot(p, t, u=nothing, pts=false)

    If `u` == nothing: Plot triangular mesh with nodes `p` and triangles `t`.
    If `u` == solution vector: Plot filled contour color plot of solution `u`.
    If 'pts', scatter verticies on grid.
"""
function tplot(p, t; u=nothing, pts=false, levels=20)
    clf()
    if u === nothing
        tripcolor(p[:, 1], p[:, 2], Array(t .- 1), 0 * t[:, 1],
            cmap="Set3", edgecolors="k", linewidth=1)
    else
        tricontourf(p[:, 1], p[:, 2], Array(t .- 1), u[:], levels)
        PyPlot.colorbar()
    end
    if pts
        plot(p[:, 1], p[:, 2], ".", markersize=18)
    end
    axis("equal")
    axis("off")
    draw()
end

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
retruns the circumcentre for a triangle of 3 points
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


part_1();
test_poisson();
part_3();
