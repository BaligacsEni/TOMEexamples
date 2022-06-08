#Example Propagator calculations of time dependent Hamiltonians for Star-Lanczos
#2022 Mai - Eni eniko.baligacs@sorbonne-universite.de
# Matrices in order 100 and 1000

using LinearAlgebra 
using Kronecker
using DifferentialEquations
# using PyPlots
using Plots

Ix = [0 0.5; 0.5 0]
Iy = [0 -0.5im; 0.5im 0]
Iz = [0.5 0; 0 -0.5 ]

function genOperatorSingleSpin(numberofspins, spinID, CBO)
    ssys = [0.5] 
    for n = 1:numberofspins
        if n == spinID 
            ssys =  ssys ⊗ CBO * 2
        else
            ssys = ssys ⊗ Matrix(I, 2,2)
        end
    end
    return ssys
end

function genOperatorDoubleSpin(numberofspins, spinID1, spinID2, CBO)
    ssys = [0.5] 
    for n = 1:numberofspins
        if n == spinID1 || n == spinID2
            ssys =  ssys ⊗ CBO * 2
        else
            ssys = ssys ⊗ Matrix(I, 2,2)
        end
    end
    return ssys
end

#
# Example D: 7 spin system (Matrix size 128 x 128). Only x + y pulses
spins = 7 
for spin in 1:spins
    name  = Symbol("s", spin, "x")
    @eval $name = $(genOperatorSingleSpin(spins, spin, Ix))
    name  = Symbol("s", spin, "y")
    @eval $name = $(genOperatorSingleSpin(spins, spin, Iy))
end


f1(t) = cos(t)
f2(t) = cos(2*t)

spinrfX = vcat(repeat([f1], spins-3), repeat([f2], 3))
spinrfY = spinrfX
MASfrequency = 10000 #in Hz
masfct(t) = (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))


RFH0(t) = genOperatorSingleSpin(spins, 1, [0 0; 0 0])
for spin in 1:spins
    rfx = Symbol("s", spin, "x")
    rfy = Symbol("s", spin, "y")
    @eval $(Symbol("RFH", spin))(t) = $(Symbol("RFH", spin-1))(t) +
    masfct(t)*(spinrfX[$spin](t)*($rfx) + spinrfY[$spin](t)*($rfy))
end
RFH(t) = eval(Symbol("RFH", spins))(t)
RFH(2)


coupvec = rand(spins, spins)
coupvec = coupvec - tril(coupvec)
# HetNcoupl = rand(Bool, spins, spins)
HetNcoupl = zeros(spins, spins)
HetNcoupl[1:4, 1:4] = (ones(4,4)- tril(ones(4,4)))
HetNcoupl[5:7, 5:7] = (ones(3,3)- tril(ones(3,3)))

Hdd0(t) = genOperatorSingleSpin(spins, 1, [0 0; 0 0])
ldd = 0
for spinA in 1:spins
    for spinB in (1+spinA):spins
        dcz = Symbol("s", spinA, spinB, "z")
        dcx = Symbol("s", spinA, spinB, "x")
        dcy = Symbol("s", spinA, spinB, "y")
        @eval $dcz = $(genOperatorDoubleSpin(spins, spinA, spinB, Iz))
        @eval $dcx = $(genOperatorDoubleSpin(spins, spinA, spinB, Ix))
        @eval $dcy = $(genOperatorDoubleSpin(spins, spinA, spinB, Iy))
        if HetNcoupl[spinA, spinB] == true
            @eval $(Symbol("Hdd", spinA, spinB))(t) = $(Symbol("Hdd", $ldd))(t) 
            #  + coupvec[$spinA, $spinB]*masfct(t)*
            #  (2*(@eval $dcz) -(@eval $dcx) -(@eval $dcy)) 
            @eval ldd = Symbol($spinA, $spinB)
            print(ldd, " ")
        else 
            @eval $(Symbol("Hdd", spinA, spinB))(t) = $(Symbol("Hdd", $ldd))(t) +
             coupvec[spinA, spinB]*masfct(t)* 2*(@eval $dcz) 
             @eval ldd = Symbol($spinA, $spinB)
            print(ldd, " ")
        end
    end
end
eval(Symbol("Hdd", spins-1, spins))(0)
Hdd67(0)

m = zeros(spins, spins)
v = 1:(cumsum(1:spins)[end])
for j in 1:spins, i in 1:spins
    if j < i 
        m[i, j] = minimum(v)



H(t) = RFD(t) + RFH(t)

Htime = [imag(H(n*dt)[1,1]) for n in 1:points]
# using Plots
timev = collect(0:dt:experimenttime)
plot((Htime), label = "Element 1,1",
    xlabel = "Points",
    ylabel = "Hamiltonian",)
savefig("ExampleAHam11.pdf")

mshape = H(0)
for n in 1:16, m in 1:16
    if mshape[n,m] == 0 
        mshape[n,m] = 0
    else mshape[n,m] = 1
    end
end

spy(mshape, marker = (:square, 10), legend = false)
savefig("MatrixshapeA.pdf")


timeOexp1 = Matrix(I, 16,16)
toevec1 = rho = Array{ComplexF64, 3}(undef, 16,16,points+1)
toevec1[:,:,1] = timeOexp1
for t = 1:points
    Ham = H(t*dt)
    prop = exp(dt*Ham)
    global timeOexp1 =  prop * timeOexp1
    toevec1[:,:,t+1] = timeOexp1
end
print("The (diagonal of the) time ordered exponential of the first exmple is:\n")
for n in 1:16
    print(round(timeOexp1[n,n], digits = 10), "\n")
end 

# timev = collect(0:dt:experimenttime)
plot([real(toevec1[1,1,n]) for n in 1:points+1], label = "Element 1,1",
    xlabel = "Points",
    ylabel = "Propagator Evolution")
savefig("ExampleAProp11.pdf")


tspan = (0, experimenttime)
LvNt(u, p, t) = H(t)*u
pro = ODEProblem(LvNt, Matrix(I, 16, 16)*(1.0+0.0im), tspan)
sol = solve(pro, abstol = 10e-8)
# display(round.(sol[end], digits = 5))
print("The (diagonal of the) time ordered exponential of the first exmple with ODEsolver is:\n")
for n in 1:16
    print(sol[end][n,n], "\n")
end 



# Example B: Non-Diagonal Matrix. 4 spins, with chemical shifts (offsets), all STRONGLY coupled. different time dependent coupling constants
# Example molecule: Furanone 13C atoms 
# generate the necessary Basis set for the Hamiltonian:
s12x = genOperatorDoubleSpin(4, 1, 2, Ix)
s13x = genOperatorDoubleSpin(4, 1, 3, Ix)
s14x = genOperatorDoubleSpin(4, 1, 4, Ix)
s23x = genOperatorDoubleSpin(4, 2, 3, Ix)
s24x = genOperatorDoubleSpin(4, 2, 4, Ix)
s34x = genOperatorDoubleSpin(4, 3, 4, Ix)

s12y = genOperatorDoubleSpin(4, 1, 2, Iy)
s13y = genOperatorDoubleSpin(4, 1, 3, Iy)
s14y = genOperatorDoubleSpin(4, 1, 4, Iy)
s23y = genOperatorDoubleSpin(4, 2, 3, Iy)
s24y = genOperatorDoubleSpin(4, 2, 4, Iy)
s34y = genOperatorDoubleSpin(4, 3, 4, Iy)

off1 = 600
off2 = 300
off3 = 0
off4 = -100

# coupling constants in Hz
couplc12 = -331.0782998347195
couplc13 = -84.27020648106176
couplc14 = -84.27020648106176
couplc23 = -358.2771424200154
couplc24 = -84.27020648106176
couplc34 = -331.0782998347195


points = 10000 #accuracy to e-9 with 10000 points!!! 
MASfrequency = 10000 #in Hz
experimenttime = 1e-3 # in s = microsecond
dt = experimenttime / points


H(t) = -im*2*pi .* (s1z*off1 + s2z*off2 + s3z*off3 + s4z*off4 +
(2*s12z-s12x-s12y) * (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplc12 +
(2*s13z-s13x-s13y) * (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplc13 +
(2*s14z-s14x-s14y) * (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplc14 +
(2*s23z-s23x-s23y) * (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplc23 +
(2*s24z-s24x-s24y) * (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplc24 +
(2*s34z-s34x-s34y) * (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplc34)

Htime = [imag(H(n*dt)[3,2]) for n in 1:points]
plot(Htime, label = "Element 3,2",
    xlabel = "Points",
    ylabel = "Hamiltonian")
savefig("ExampleBHam32.pdf")

# plot([imag(H(n*dt)[1,1]) for n in 1:points])
mshape = H(0)
for n in 1:16, m in 1:16
    if mshape[n,m] == 0 
        mshape[n,m] = 0
    else mshape[n,m] = 1
    end
end

spy(mshape, marker = (:square, 10), legend = false)
savefig("MatrixshapeB.pdf")

timeOexp2 = Matrix(I, 16,16)
toevec2 = rho = Array{ComplexF64, 3}(undef, 16,16,points+1)
toevec2[:,:,1] = timeOexp2
for t = 1:points
    Ham = H(t*dt)
    prop = exp(dt*Ham)
    global timeOexp2 =  prop * timeOexp2
    toevec2[:,:,t+1] = timeOexp2
end
print("The time ordered exponential of the second exmple is:\n")
display(round.(timeOexp2, digits = 10))

plot([real(toevec2[3,2,n]) for n in 1:points], label = "Element 3,2",
    xlabel = "Points",
    ylabel = "Propagator Evolution")
savefig("ExampleBProp32.pdf")

tspan = (0, experimenttime)
LvNt(u, p, t) = H(t)*u
pro = ODEProblem(LvNt, Matrix(I, 16, 16)*(1.0+0.0im), tspan)
sol = solve(pro, abstol = 10e-8)
print("The time ordered exponential of the second exmple with ODEsolver is:\n")
display(round.(sol[end], digits = 10))


# toe = [0.0+0.0im for n in 1:points+1]
# toe[1] = 1
# experimenttime = 1e-3 # in s = microsecond
# dt = experimenttime / points
# for t in 1:points
#     prop = exp(dt*H(t*dt)[1,1])
#     print(H(t*dt)[1,1], "\n")
#     toe[t+1] = prop * toe[t]
# end

# plot(real(toe))




#
# Example C: 4 spins under a shaped pulse
# generate the necessary Basis set for the Hamiltonian:

s1x = genOperatorSingleSpin(4, 1, Ix)
s2x = genOperatorSingleSpin(4, 2, Ix)
s3x = genOperatorSingleSpin(4, 3, Ix)
s4x = genOperatorSingleSpin(4, 4, Ix)

s1y = genOperatorSingleSpin(4, 1, Iy)
s2y = genOperatorSingleSpin(4, 2, Iy)
s3y = genOperatorSingleSpin(4, 3, Iy)
s4y = genOperatorSingleSpin(4, 4, Iy)

# random shaped pulse:
H(t) = -im*2*pi .* (s1z*off1 + s2z*off2 + s3z*off3 + s4z*off4 +
    (0.5 + cos(40*t) + sin(100*t) - 0.4*sin(160*t/points))*100 *(s1x+ s2x+ s3x+ s4x)+
    (sin(40*t) + cos(80*t) + 2*sin(120*t))                *100 *(s1y+ s2y+ s3y+ s4y))

experimenttime = 1e-1
dt = experimenttime/points

Htime = [imag(H(n*dt)[5,1]) for n in 1:points]
plot(Htime, label = "Element 5,1",
    xlabel = "Points",
    ylabel = "Hamiltonian")
savefig("ExampleCHam51.pdf")


timeOexp3 = Matrix(I, 16,16)
toevec3 = rho = Array{ComplexF64, 3}(undef, 16,16,points+1)
toevec3[:,:,1] = timeOexp3
for t = 1:points
    Ham = H(t*dt)
    prop = exp(dt*Ham)
    global timeOexp3 =  prop * timeOexp3 
    toevec3[:,:,t+1] = timeOexp3
end
print("The time ordered exponential of the third exmple is:\n")
display(round.(timeOexp3, digits = 10))

# tspan = (0, experimenttime)
# LvNt(u, p, t) = H(t)*u
# pro = ODEProblem(LvNt, Matrix(I, 16, 16)*(1.0+0.0im), tspan)
# sol = solve(pro, abstol = 10e-8)
# print("The time ordered exponential of the third exmple with ODEsolver is:\n")
# display(round.(sol[end], digits = 10))

mshape = H(0)
for n in 1:16, m in 1:16
    if mshape[n,m] == 0 
        mshape[n,m] = 0
    else mshape[n,m] = 1
    end
end

spy(mshape, marker = (:square, 10), legend = false)
savefig("MatrixshapeC.pdf")

plot([real(toevec3[5,1,n]) for n in 1:points], label = "Element 5,1",
    xlabel = "Points",
    ylabel = "Propagator Evolution")
savefig("ExampleCProp51.pdf")


#=
Distnaces between atoms in RNA:
Example from: PDB 2KOC
Atoms used here:
                                   x        y      z
ATOM    196  P     U A   7      12.839  -1.509  -3.823  1.00 10.00           P  
ATOM    201  C4'   U A   7      15.293  -0.888  -0.888  1.00 10.00           C 
ATOM    208  N1    U A   7      14.368   2.519  -1.295  1.00 10.00           N 
ATOM    225  H6    U A   7      13.220   1.262  -2.580  1.00 10.00           H 
measure distances according:
=#

# distances in Angstrom
rPC = sqrt((12.839-15.293)^2 + (-1.509 +0.888)^2 + (-3.823 +0.888)^2)*1e-10
rPN = sqrt((12.839-14.368)^2 + (-1.509 -2.519)^2 + (-3.823 +1.295)^2)*1e-10
rPH = sqrt((12.839-13.220)^2 + (-1.509 -1.262)^2 + (-3.823 +2.580)^2)*1e-10

rCN = sqrt((15.293-14.368)^2 + (-0.888-2.519 )^2 + (-0.888+1.295)^2)*1e-10
rCH = sqrt((15.293-13.220)^2 + (-0.888-1.262 )^2 + (-0.888+2.580)^2)*1e-10

rNH = sqrt((14.368- 13.220 )^2 + (2.519-1.262 )^2 + (-1.295+2.580)^2)*1e-10

# gyromagnetic ratio Hz/T
gH = 42.577e6 
gC = 10.708e6
gN =  4.316e6 # actually negative but the equation uses absolut values
gP = 17.235e6 

μ0 = 4*pi*1e-7 # Vacuum permeability
h = 1.0545718e−34 #Plancks constant

couplcPC = -μ0*gP*gC*h/(4*pi*rPC^3)
couplcPN = -μ0*gP*gN*h/(4*pi*rPN^3)
couplcPH = -μ0*gP*gH*h/(4*pi*rPH^3)
couplcCN = -μ0*gC*gN*h/(4*pi*rCN^3)
couplcCH = -μ0*gC*gH*h/(4*pi*rCH^3)
couplcNH = -μ0*gN*gH*h/(4*pi*rNH^3)

r12 = 154e-12
r13 = 243e-12
r14 = 243e-12
r23 = 150e-12
r24 = 243e-12
r34 = 154e-12

# Homonuclear case for 13C 
couplc12 = -μ0*gC^2*h/(4*pi*r12^3)
couplc13 = -μ0*gC^2*h/(4*pi*r13^3)
couplc14 = -μ0*gC^2*h/(4*pi*r14^3)
couplc23 = -μ0*gC^2*h/(4*pi*r23^3)
couplc24 = -μ0*gC^2*h/(4*pi*r24^3)
couplc34 = -μ0*gC^2*h/(4*pi*r34^3)


