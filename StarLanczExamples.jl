#Example Propagator calculations of time dependent Hamiltonians for Star-Lanczos
#2022 January - Eni eniko.baligacs@sorbonne-universite.de
#
using LinearAlgebra 
using Kronecker
using DifferentialEquations

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
# Example A: Diagonal Matrix. 4 spins, with chemical shifts (offsets), all weakly coupled. different time dependent coupling constants 
# generate the necessary Basis set for the Hamiltonian:
s1z = genOperatorSingleSpin(4, 1, Iz)
s2z = genOperatorSingleSpin(4, 2, Iz)
s3z = genOperatorSingleSpin(4, 3, Iz)
s4z = genOperatorSingleSpin(4, 4, Iz)

s12z = genOperatorDoubleSpin(4, 1, 2, Iz)
s13z = genOperatorDoubleSpin(4, 1, 3, Iz)
s14z = genOperatorDoubleSpin(4, 1, 4, Iz)
s23z = genOperatorDoubleSpin(4, 2, 3, Iz)
s24z = genOperatorDoubleSpin(4, 2, 4, Iz)
s34z = genOperatorDoubleSpin(4, 3, 4, Iz)

#calculate the time ordered exponential: propagator = exp(-i*dt*Hamiltonian)
off1H = 420e6
off13C = -105e6
off15N = 350e6
off31P = 0

experimenttime = 100e-6 # in s = microsecond
points = 1000 #accuracy to e-9 with 10000 points!!! 
dt = experimenttime / points
# coupling constants in Hz
couplcPC = -33.4275808727199
couplcPN = -6.29323646000687
couplcPH = -269.865214403666
couplcCN = -10.8596662768044
couplcCH = -118.875602347998
couplcNH = -199.725378119845


MASfrequency = 10000 #in Hz
masperTime = experimenttime * MASfrequency
timeOexp1 = Matrix(I, 16,16)

H(t) = -im*2*pi .* (s1z*off1H + s2z*off13C + s3z*off15N + s4z*off31P +
2*s12z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplcCH +
2*s13z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplcNH +
2*s14z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplcPH +
2*s23z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplcCN +
2*s24z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplcPC +
2*s34z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* couplcPN)

for t = 1:points
    Ham = H(t*dt)
    prop = exp(dt*Ham)
    global timeOexp1 =  prop * timeOexp1
end
print("The (diagonal of the) time ordered exponential of the first exmple is:\n")
for n in 1:16
    print(timeOexp1[n,n], "\n")
end 

tspan = (0, experimenttime)
LvNt(u, p, t) = H(t)*u
pro = ODEProblem(LvNt, s1z+s2z*(1+0im), tspan)
sol = solve(pro, abstol = 10e-8)
display(round.(sol[end], digits = 5))



#
# Example B: Non-Diagonal Matrix. 4 spins, with chemical shifts (offsets), all STRONGLY coupled. different time dependent coupling constants
# Example molecule: Furanone 14C atoms 
# generate the necessary Basis set for the Hamiltonian:

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

timeOexp2 = Matrix(I, 16,16)
for t = 1:points
    Ham = s1z*off1 + s2z*off2 + s3z*off3 + s4z*off4 +
        (2*s12z-s12x-s12y) *cos(2*pi*masperTime*t/points)* couplc12 +
        (2*s13z-s13x-s13y) *cos(2*pi*masperTime*t/points)* couplc13 +
        (2*s14z-s14x-s14y) *cos(2*pi*masperTime*t/points)* couplc14 +
        (2*s23z-s23x-s23y) *cos(2*pi*masperTime*t/points)* couplc23 +
        (2*s24z-s24x-s24y) *cos(2*pi*masperTime*t/points)* couplc24 +
        (2*s34z-s34x-s34y) *cos(2*pi*masperTime*t/points)* couplc34
    prop = exp(dt*-im*Ham)
    global timeOexp2 =  prop * timeOexp2
end
print("The time ordered exponential of the second exmple is:\n")
display(round.(real.(timeOexp2), digits = 1))



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

timeOexp3 = Matrix(I, 16,16)
for t = 1:points
    Ham = s1z*off1 + s2z*off2 + s3z*off3 + s4z*off4 +
    (0.5 + cos(4*t/points) + sin(10*t/points) - 0.4*sin(16*t/points))*100 *(s1x+ s2x+ s3x+ s4x)+
    (sin(4*t/points) + cos(8*t/points) + 2*sin(12*t/points))         *100 *(s1y+ s2y+ s3y+ s4y)
    prop = exp(dt*-im*Ham)
    global timeOexp3 =  prop * timeOexp3 
end
print("The time ordered exponential of the second exmple is:\n")
display(round.(timeOexp3, digits = 9))



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


