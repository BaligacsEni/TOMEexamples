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
off1 = 0.2
off2 = 0.1
off3 = 0
off4 = -0.2

experimenttime = 100e-6 # in s = microsecond
points = 1000 #accuracy to e-9 with 10000 points!!! 
dt = experimenttime / points
weakcc12 = 10000 #in Hz 
weakcc13 = 20000 #in Hz 
weakcc14 = 5000 #in Hz 
weakcc23 = 12000 #in Hz 
weakcc24 = 14000 #in Hz 
weakcc34 = 8000 #in Hz 
MASfrequency = 10000 #in Hz
masperTime = experimenttime * MASfrequency
timeOexp1 = Matrix(I, 16,16)

H(t) = -im*2*pi .* (s1z*off1 + s2z*off2 + s3z*off3 + s4z*off4 +
s12z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* weakcc12 +
s13z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* weakcc13 +
s14z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* weakcc14 +
s23z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* weakcc23 +
s24z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* weakcc24 +
s34z* (cos(2*pi*MASfrequency*t) + cos(4*pi*MASfrequency*t))* weakcc34)

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

timeOexp2 = Matrix(I, 16,16)
for t = 1:points
    Ham = s1z*off1 + s2z*off2 + s3z*off3 + s4z*off4 +
        (2*s12z-s12x-s12y) *cos(2*pi*masperTime*t/points)* weakcc12 +
        (2*s13z-s13x-s13y) *cos(2*pi*masperTime*t/points)* weakcc13 +
        (2*s14z-s14x-s14y) *cos(2*pi*masperTime*t/points)* weakcc14 +
        (2*s23z-s23x-s23y) *cos(2*pi*masperTime*t/points)* weakcc23 +
        (2*s24z-s24x-s24y) *cos(2*pi*masperTime*t/points)* weakcc24 +
        (2*s34z-s34x-s34y) *cos(2*pi*masperTime*t/points)* weakcc34
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
