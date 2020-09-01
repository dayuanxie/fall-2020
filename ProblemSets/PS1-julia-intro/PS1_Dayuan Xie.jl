# PS1 Dayuan Xie

# using Pkg
# Pkg.add("JLD2") 
# Pkg.add("Random") 
# Pkg.add("LinearAlgebra") 
# Pkg.add("Statistics") 
# Pkg.add("CSV") 
# Pkg.add("DataFrames") 
# Pkg.add("FreqTables") 
# Pkg.add("Distributions") 
# Pkg.add("seed")
# Pkg.add("Revise") 
# Pkg.add("HTTP") 
# Pkg.add("FreqTables") 


# Question 1 (a)
using Random
using Distributions
Random.seed!(1234)
A=rand(Uniform(-5,10),10,7)
B=rand(Normal(-2,15),10,7)
C=[A[1:5,1:5] B[1:5,6:7]]
D=copy(A)
D[D.>0] .=0
println(D)


# Question 1 (b)
length(A)


# Question 1 (c)
length(setdiff(D))


# Question 1 (d)
E=reshape(B,70,1).*reshape(B,1,70)


# Question 1 (e)
F = cat(A, B, dims=3) 


# Question 1 (f)
F = permutedims(F, [3, 1, 2]) 


# Question 1 (g)
G=kron(B,C)
# G=kron(C,F) Error


# Question 1 (h)
using JLD2
file = jldopen("D:\\julia\\metrics\\ps1\\output\\matrixpractice.jld", "w")
write(file, "A", A)
write(file, "B", B)
write(file, "C", C)
write(file, "D", D)
write(file, "E", E)
write(file, "F", F)
write(file, "G", G)
close(file)
# file= jldopen("D:\\julia\\metrics\\ps1\\output\\matrixpractice.jld", "r")
# h = read(file, "A")
# println(h==A)
# if h = A ? Yes.


# Question 1 (i)
file = jldopen("D:\\julia\\metrics\\ps1\\output\\firstmatrix.jld", "w")
write(file, "A", A)
write(file, "B", B)
write(file, "C", C)
write(file, "D", D)
close(file)


# Question 1 (j)
using DataFrames
using CSV
data=convert(DataFrame, C)
CSV.write("D:\\julia\\metrics\\ps1\\output\\Cmatrix.csv", data)
# CSV.write("D:\\julia\\metrics\\ps1\\output\\Cmatrix.csv", DataFrame(C))


# Question 1 (k)
data=convert(DataFrame, D)
CSV.write("D:\\julia\\metrics\\ps1\\output\\Dmatrix.dat", data)
# D=CSV.read("D:\\julia\\metrics\\ps1\\output\\Dmatrix.dat")
# print(D)


# Question 1 (l)
using Random, Distributions
function q1()
Random.seed!(1234)
A=rand(Uniform(-5,10),10,7)
B=rand(Normal(-2,15),10,7)
C=[A[1:5,1:5] B[1:5,6:7]]
D=copy(A)
D[D.>0] .=0
    return A, B, C, D
end

A,B,C,D = q1()



# Question 2 (a)
AB=[i for i in vec(A)]*[j for j in vec(B)]'
AB2=vec(A) * vec(B)'


# Question 2 (b)
Cprime=vec(C)
for i in Cprime
    if i<-5  
       filter!(x->x!=i,Cprime)
    end
    for i in Cprime
     if i>5
       filter!(y->y!=i,Cprime)
     end
    end
end
print(Cprime)

Cprime2=[i for i in vec(C) if i>=-5 && i<=5]


# Question 2 (c)
using Distributions
K1=ones(Float64,15169*5,1)
K2=zeros(Float64,15169*5,1)
K3=zeros(Float64,15169*5,1)
K4=zeros(Float64,15169*5,1)
K5=zeros(Float64,15169*5,1)
K6=zeros(Float64,15169*5,1)
for t in 1:5
  #K2[1:1569t]=bitrand([0.75∗(6−t)//5],15169,1)
  K3[1:1569t,1]=rand(Normal(15+t-1,5(t-1)),15169,1)
  K4[1:1569t,1]=rand(Normal(pi(6-t)//3,1//exp(1)),15169,1)
  K5[1:1569t,1]=rand(Binomial(20,0.6),15169,1)
  K6[1:1569t,1]=rand(Binomial(20,0.5),15169,1)
end

X = cat(K1,K2,K3,K4,K5,K6, dims=5) 


# Question 2 (d)
β=zeros(Float64,6,5)
β1=collect(1:0.25:2)
for t in 1:5
    β[1,t]=β1[t,1]
    β[2,t]=log(t)
    β[3,t]=-sqrt(t)
    β[4,t]=exp(t)-exp(t+1)
    β[5,t]=t
    β[6,t]=t//3
end
print(β)


# Question 2 (e)
ε=rand(Normal(0,0.36),15169,5)
Y=zeros(Float64,15169,5)
for n in 1:15169
    for t in 1:5
        for k in 1:6
        Y[n,t]=X[n,k,t]β[k,t]+ε[n,t]
        end
    end
end

print(Y)


# Question 2 (f)
using Random, Distributions
function q2(A,B,C)
A,B,C,D = q1()
AB=[i for i in vec(A)]*[j for j in vec(B)]'
AB2=vec(A) * vec(B)'
Cprime=vec(C)
for i in Cprime
    if i<-5  
       filter!(x->x!=i,Cprime)
    end
    for i in Cprime
     if i>5
       filter!(y->y!=i,Cprime)
     end
    end
end
print(Cprime)
Cprime2=[i for i in vec(C) if i>=-5 && i<=5]
K1=ones(Float64,15169*5,1)
K2=zeros(Float64,15169*5,1)
K3=zeros(Float64,15169*5,1)
K4=zeros(Float64,15169*5,1)
K5=zeros(Float64,15169*5,1)
K6=zeros(Float64,15169*5,1)
for t in 1:5
  #K2[1:1569t]=bitrand([0.75∗(6−t)//5],15169,1)
  K3[1:1569t,1]=rand(Normal(15+t-1,5(t-1)),15169,1)
  K4[1:1569t,1]=rand(Normal(pi(6-t)//3,1//exp(1)),15169,1)
  K5[1:1569t,1]=rand(Binomial(20,0.6),15169,1)
  K6[1:1569t,1]=rand(Binomial(20,0.5),15169,1)
end
X = cat(K1,K2,K3,K4,K5,K6, dims=5)  
β=zeros(Float64,6,5)
β1=collect(1:0.25:2)
for t in 1:5
    β[1,t]=β1[t,1]
    β[2,t]=log(t)
    β[3,t]=-sqrt(t)
    β[4,t]=exp(t)-exp(t+1)
    β[5,t]=t
    β[6,t]=t//3
end
print(β)
ε=rand(Normal(0,0.36),15169,5)
Y=zeros(Float64,15169,5)
for n in 1:15169
    for t in 1:5
        for k in 1:6
        Y[n,t]=X[n,k,t]β[k,t]+ε[n,t]
        end
    end
end

print(Y)
end

q2(A,B,C)



# Question 3 (a)
using CSV, HTTP, DataFrames,JLD2
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
nlsw88 = CSV.read(HTTP.get(url).body)
file = jldopen("D:\\julia\\metrics\\ps1\\output\\nlsw88.jld", "w")


# Question 3 (b)
using Statistics
mean(nlsw88.never_married)
mean(nlsw88.collgrad)


# Question 3 (c)
# tabulate
# using FreqTables
# freqtable(x)
# tabulate(nlsw88.race)


# Question 3 (d)
summarystats= describe(nlsw88,:mean, :median, :std, :min, :max,:nunique,:q25,:q75,:nmissing)


# Question 3 (e)
using FreqTables
freqtable(nlsw88.industry, nlsw88.occupation)


# Question 3 (f)
nlsw88_f=select(nlsw88, :industry,:occupation,:wage)
gdf = groupby(nlsw88_f, [:industry,:occupation])
combine(gdf, :wage => mean)
#by(nlsw88_f, [:industry,:occupation], :wage => mean)


# Question 3 (g)
using CSV, HTTP, DataFrames,JLD2,Statistics,FreqTables
function q3()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    nlsw88 = CSV.read(HTTP.get(url).body)
    file = jldopen("D:\\julia\\metrics\\ps1\\output\\nlsw88.jld", "w")
    mean(nlsw88.never_married)
    mean(nlsw88.collgrad)
    summarystats= describe(nlsw88,:mean, :median, :std, :min, :max,:nunique,:q25,:q75,:nmissing)
    freqtable(nlsw88.industry, nlsw88.occupation)
    nlsw88_f=select(nlsw88, :industry,:occupation,:wage)
    gdf = groupby(nlsw88_f, [:industry,:occupation])
    combine(gdf, :wage => mean)
end
q3()




# Question 4 (a)
using JLD2
file = jldopen("D:\\julia\\metrics\\ps1\\output\\firstmatrix.jld", "r")
A = read(file, "A")
B = read(file, "B")
C = read(file, "C")
D = read(file, "D")


# Question 4 (b)
function  matrixops(X,Y)
    a=[i for i in vec(X)]*[j for j in vec(Y)]'
    b= X'*Y
    c=sum(X.+Y)
    println(a) 
    println(b)
    println(c)
end
# using JLD2
# function  matrixops()
#     file = jldopen("D:\\julia\\metrics\\ps1\\output\\firstmatrix.jld", "r")
#     A = read(file, "A")
#     B = read(file, "B")
#     a=[i for i in A]*[j for j in B]'
#     b= A'*B
#     c=sum(A.+B)
#     println(a) 
#     println(b)
#     println(c)
# end
# matrixops()


# Question 4 (c)
function  matrixops(X,Y)
    #  Calculate the element-by-element product of X and Y
    a=[i for i in vec(X)]*[j for j in vec(Y)]'
    #  Calculate the product of X'Y
    b= X'*Y
    # Calculate the sum of all the elements of X+Y
    c=sum(X.+Y)
    # Print results
    println(a) 
    println(b)
    println(c)
end


# Question 4 (d)
using JLD2
file = jldopen("D:\\julia\\metrics\\ps1\\output\\firstmatrix.jld", "r")
A = read(file, "A")
B = read(file, "B")
matrixops(A,B)


# Question 4 (e)
function  matrixops(X,Y)
    #  Calculate the element-by-element product of X and Y
    if  size(X)==size(Y)
        println([i for i in vec(X)]*[j for j in vec(Y)]')
    else
        println("inputs must have the same size")
    end
    #  Calculate the product of X'Y
    b= X'*Y
    # Calculate the sum of all the elements of X+Y
    c=sum(X.+Y)
    # Print results
    println(b)
    println(c)
end


# Question 4 (f)
using JLD2
file = jldopen("D:\\julia\\metrics\\ps1\\output\\firstmatrix.jld", "r")
C = read(file, "C")
D = read(file, "D")
matrixops(C,D)


# Question 4 (g)
using CSV, HTTP, DataFrames,JLD2
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
nlsw88 = CSV.read(HTTP.get(url).body)
E=convert(Array,nlsw88.ttl_exp)
F=convert(Array,nlsw88.wage)
matrixops(E,F)


# Question 4 (h)
using CSV, HTTP, DataFrames,JLD2
function q4()
    file = jldopen("D:\\julia\\metrics\\ps1\\output\\firstmatrix.jld", "r")
    A = read(file, "A")
    B = read(file, "B")
    C = read(file, "C")
    D = read(file, "D")
    function  matrixops(X,Y)
        #  Calculate the element-by-element product of X and Y
        a=[i for i in vec(X)]*[j for j in vec(Y)]'
        #  Calculate the product of X'Y
        b= X'*Y
        # Calculate the sum of all the elements of X+Y
        c=sum(X.+Y)
        # Print results
        println(a) 
        println(b)
        println(c)
    end
    matrixops(A,B)
    function  matrixops(X,Y)
        #  Calculate the element-by-element product of X and Y
        if  size(X)==size(Y)
            println([i for i in vec(X)]*[j for j in vec(Y)]')
        else
            println("inputs must have the same size")
        end
        #  Calculate the product of X'Y
        b= X'*Y
        # Calculate the sum of all the elements of X+Y
        c=sum(X.+Y)
        # Print results
        println(b)
        println(c)
    end
    matrixops(C,D)
    
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    nlsw88 = CSV.read(HTTP.get(url).body)
    E=convert(Array,nlsw88.ttl_exp)
    F=convert(Array,nlsw88.wage)
    matrixops(E,F)
    
end
q4()







