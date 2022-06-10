#Example Propagator calculations of time dependent Hamiltonians for Star-Lanczos
#2022 Mai - Eni eniko.baligacs@sorbonne-universite.de
# Matrices in order 100 and 1000

using LinearAlgebra 
using Kronecker
using DifferentialEquations
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


function getHamShape(H) 
    mshape = H(0)
    ms = size(mshape)[1]
    for n in 1:ms, m in 1:ms
        if mshape[n,m] == 0 
            mshape[n,m] = 0
        else mshape[n,m] = 1
        end
    end
    return mshape
end

function getPropShape(U)
    mshape = U
    ms = size(mshape)[1]
    for n in 1:ms, m in 1:ms
        if mshape[n,m] == 0 
            mshape[n,m] = 0
        else mshape[n,m] = 1
        end
    end
    return mshape
end

###########################################################
# todo: test functions!
# Example D: 7 spin system (Matrix size 128 x 128). Only x + y pulses
spins = 5
exptime = 0.01
nt = 100
dt = exptime/nt

# generate Operators for the x/y pulses
for spin in 1:spins
    name  = Symbol("s", spin, "x")
    @eval $name = $(genOperatorSingleSpin(spins, spin, Ix))
    name  = Symbol("s", spin, "y")
    @eval $name = $(genOperatorSingleSpin(spins, spin, Iy))
end

# functions for shaped pulse
fx1(t) = cos(1000*t*2*pi) 
fx2(t) = cos(1500*t*2*pi)
fy1(t) = sin(1000*t*2*pi) 
fy2(t) = sin(1500*t*2*pi)
spinrfX = vcat(repeat([fx1], spins-3), repeat([fx2], 3))
spinrfY = vcat(repeat([fy1], spins-3), repeat([fy2], 3))

# test the shaped pulse functios
plot([fx1(n*dt) for n in 1:nt])

# Magic angle spinning frequency + its modulation
MASfrequency = 1000 #in Hz
masfct(t) = (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))
# test masfunction
plot([masfct(n*dt) for n in 1:nt])


# generate the full Hamiltonian of all spins with 
RFH0(t) = genOperatorSingleSpin(spins, 1, [0 0; 0 0])
for spin in 1:spins
    rfx = Symbol("s", spin, "x")
    rfy = Symbol("s", spin, "y")
    @eval $(Symbol("RFH", spin))(t) = $(Symbol("RFH", spin-1))(t) +
    masfct(t)*(spinrfX[$spin](t)*($rfx) + spinrfY[$spin](t)*($rfy))
end
RFH(t) = eval(Symbol("RFH", spins))(t)

# see shape of the full Radiofrequency Hamiltonian
spy(getHamShape(RFH), marker = (:square, 6))

# test single elements
plot([imag(RFH(n*dt)[1,5]) for n in 1:nt])


##############create the dipolar coupling Hamiltonian
coupvec = rand(spins, spins)*2*pi*1000 #in Hz
coupvec = coupvec - tril(coupvec)
# Homcoupl = rand(Bool, spins, spins)
# here: heteronuclear coupling only between the first 4 vs last 3
Homcoupl = zeros(spins, spins)
Homcoupl[1:3, 1:3] = (ones(3,3)- tril(ones(3,3)))
Homcoupl[4:5, 4:5] = (ones(2,2)- tril(ones(2,2)))



Hdd0(t) = genOperatorSingleSpin(spins, 1, [0 0; 0 0])
c = 1
for spinA in 1:spins
    for spinB in (1+spinA):spins
        dcz = Symbol("s", spinA, spinB, "z")
        dcx = Symbol("s", spinA, spinB, "x")
        dcy = Symbol("s", spinA, spinB, "y")
        @eval $dcz = $(genOperatorDoubleSpin(spins, spinA, spinB, Iz))
        @eval $dcx = $(genOperatorDoubleSpin(spins, spinA, spinB, Ix))
        @eval $dcy = $(genOperatorDoubleSpin(spins, spinA, spinB, Iy))
        if Homcoupl[spinA, spinB] == true
            @eval $(Symbol("Hdd", c))(t) = $(Symbol("Hdd", c-1))(t) +
            coupvec[$spinA, $spinB]*masfct(t)*(2*(@eval $dcz) -(@eval $dcx) -(@eval $dcy)) 
            c +=1
        else 
            @eval $(Symbol("Hdd", c))(t) = $(Symbol("Hdd", c-1))(t) +
            coupvec[$spinA, $spinB]*masfct(t)*2*(@eval $dcz) 
            c +=1
        end
    end
end
@eval $(Symbol("Hdd"))(t) = $(Symbol("Hdd", c-1))(t)

# see shape of the full dipolar coupling Hamiltonian
spy(getHamShape(Hdd), marker = (:square, 6))

# test single elements
plot([real(Hdd(n*dt)[3,2]) for n in 1:nt])




#####################
# the full Hamiltonian is the Hamiltonian of the pulses and the coupling
H(t) = Hdd(t) + RFH(t)

# see shape of the full Hamiltonian
spy(getHamShape(RFH), marker = (:square, 6))

# test single elements
plot([real(H(n*dt)[1,]) for n in 1:nt])





########### solve with suzuki-trotter
ms = size(H(0))[1]
timeOexp1 = Matrix(I, ms,ms)
toevec1 = Array{ComplexF64, 3}(undef, ms,ms,nt+1)
toevec1[:,:,1] = timeOexp1
for t = 1:nt
    Ham = H(t*dt)
    prop = exp(dt*Ham)
    global timeOexp1 =  prop * timeOexp1
    toevec1[:,:,t+1] = timeOexp1
end

plot([real(toevec1[1,1,n]) for n in 1:nt+1], label = "Element 1,1",
    xlabel = "Points",
    ylabel = "Propagator Evolution")
spy(getPropShape(toevec1[:,:,2]), marker = (:square, 1.5))

############# solve with solver
# if the solver still works with your matrix size: (good luck!)
tspan = (0, exptime)
LvNt(u, p, t) = H(t)*u
pro = ODEProblem(LvNt, Matrix(I, ms, ms)*(1.0+0.0im), tspan)
sol = solve(pro, abstol = 10e-8)


