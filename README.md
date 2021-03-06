<img src="https://i.imgur.com/8ycOn14.png" height="120">

| **Documentation** | **Build Status** | **References to cite** |
|:-----------------:|:----------------:|:-----------------------:|
| [![][docs-stable-img]][docs-stable-url] | [![Build Status][build-img]][build-url] | [![DOI](https://zenodo.org/badge/99226144.svg)](https://zenodo.org/badge/latestdoi/99226144) |
| [![][docs-latest-img]][docs-latest-url] | [![Codecov branch][codecov-img]][codecov-url] | [arXiv:1711.10911](https://arxiv.org/abs/1711.10911) |


`HomotopyContinuation.jl` is a package for solving polynomial systems with only finitely many solution using numerical homotopy continuation.

You can simply install this package via the Julia package manager
```julia
julia> Pkg.add("HomotopyContinuation");
```

For more information and guides visit [www.JuliaHomotopyContinuation.org](http://www.JuliaHomotopyContinuation.org).

## A first example
HomotopyContinuation.jl aims at having easy-to-understand top-level commands. For instance, suppose we want to solve the following system

```math
f= [x^2+y, y^2-1].  
```
This can be accomplished as follows
```julia
julia> using HomotopyContinuation

julia> @polyvar x y; # declare the variables x and y

julia> result = solve([x^2+2y, y^2-2])
-----------------------------------------------
Paths tracked: 4
# non-singular finite solutions:  4
# singular finite solutions:  0
# solutions at infinity:  0
# failed paths:  0
Random seed used: 622483
-----------------------------------------------
```
Let us see what is the information that we get. Four paths were attempted to be solved, four of which were completed successfully. Since we tried to solve an affine system, the algorithm checks whether there are solutions at infinity: in this case there are none. With *solutions at infinity* we mean solutions of the [homogenization](https://en.wikipedia.org/wiki/Homogeneous_polynomial#Homogenization) of the system which are no solutions of the affine system. None of the solutions are singular and two of them are real.

Assume we are only interested in the *real* solutions. We can obtain these `result` by
```julia
julia> real(result)
2-element Array{HomotopyContinuation.Solving.PathResult{Complex{Float64},Float64,Complex{Float64}},1}:
 * returncode: success
 * solution: Complex{Float64}[1.68179+1.27738e-17im, -1.41421-1.18454e-17im]
 * residual: 4.454e-16

 * returncode: success
 * solution: Complex{Float64}[-1.68179+1.104e-18im, -1.41421+2.03235e-18im]
 * residual: 8.882e-16
```
where we can see that there are 2 real solutions, `(⁴√(2³),-√2)` and `(-⁴√(2³), -√2)`. We also can look into more detail into the first result by
```julia
julia> real(result)[1]
 ---------------------------------------------
 * returncode: success
 * solution: Complex{Float64}[1.68179+1.27738e-17im, -1.41421-1.18454e-17im]
 * residual: 4.454e-16
 * condition_number: 1.933e+00
 * windingnumber: 1
 ----
 * path number: 3
 * start_solution: Complex{Float64}[1.0+0.0im, -1.0+1.22465e-16im]
 ----
 * t: 0.0
 * iterations: 4
 * npredictions: 1
 ---------------------------------------------
```

The returncode tells us that the pathtracking was successfull. What do the other entries of that table tell us? Let us consider the most relevant.
- `solution`: the zero that is computed (here it is `(⁴√(2³),-√2)`).
- `singular`: boolean that shows whether the zero is singular.
- `residual`: the computed value of ``|f([1.68179+1.27738e-17im, -1.41421-1.18454e-17im])|``.
- `windingnumber`: This is a lower bound on the multiplicity of the solution. A windingnumber greater than 1 indicates that the solution is singular.

To extract the solutions you can do
```julia
julia> results(solution, result)
4-element Array{Array{Complex{Float64},1},1}:
 Complex{Float64}[4.44089e-15+1.68179im, 1.41421-8.88178e-16im]
 Complex{Float64}[1.86717e-16-1.68179im, 1.41421-6.50354e-17im]
 Complex{Float64}[1.68179+1.27738e-17im, -1.41421-1.18454e-17im]
 Complex{Float64}[-1.68179+1.104e-18im, -1.41421+2.03235e-18im]
```
or
```julia
results(solution, result, onlyreal=true)
2-element Array{Array{Complex{Float64},1},1}:
 Complex{Float64}[1.68179+1.27738e-17im, -1.41421-1.18454e-17im]
 Complex{Float64}[-1.68179+1.104e-18im, -1.41421+2.03235e-18im]
```
to obtain only real solutions.
See the documentation for more possibilities to analyze `result`.

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-stable-url]: https://www.juliahomotopycontinuation.org/HomotopyContinuation.jl/stable
[docs-latest-url]: https://www.juliahomotopycontinuation.org/HomotopyContinuation.jl/latest

[build-img]: https://travis-ci.org/JuliaHomotopyContinuation/HomotopyContinuation.jl.svg?branch=master
[build-url]: https://travis-ci.org/JuliaHomotopyContinuation/HomotopyContinuation.jl
[codecov-img]: https://codecov.io/gh/juliahomotopycontinuation/HomotopyContinuation.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/juliahomotopycontinuation/HomotopyContinuation.jl
