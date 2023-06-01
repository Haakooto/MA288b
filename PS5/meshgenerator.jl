using PyCall
using Plots, TriplotRecipes
using SparseArrays, LinearAlgebra

""" Meshgenerator """

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
