using Knet,Images; 
include(Pkg.dir("Knet","data","mnist.jl"))
global atype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}

#A generic MLP function with customizable activation functions 
function mlp(w,x;p=0.0,activation=leakyrelu,outputactivation=sigm)
    for i=1:2:length(w)
        x = w[i]*dropout(mat(x),p) .+ w[i+1]   # mat() used for flatten images to vector.
        i<length(w)-1 && (x = activation.(x)) 
    end
    return outputactivation.(x) #output layer
end
leakyrelu(x;α=Float32(0.2)) = max(0,x) + α*min(0,x) # LeakyRelu activation

global const 𝜀=Float32(1e-8) #  a small number prevent from getting NaN  in logs
D(w,x;p=0.0) = mlp(w,x;p=p)  #  Discriminator
G(w,z;p=0.0) = mlp(w,z;p=p)  #  Generator
𝑱d(𝗪d,x,Gz) = -mean(log.(D(𝗪d,x)+𝜀)+log.(1-D(𝗪d,Gz)+𝜀))/2 # Discriminator Loss
𝑱g(𝗪g, 𝗪d, z) = -mean(log.(D(𝗪d,G(𝗪g,z))+𝜀))             # Generator Loss
𝒩(input, batch) = atype(randn(Float32, input, batch))      # SampleNoise 

∇d  = grad(𝑱d) # Discriminator gradient
∇g  = grad(𝑱g) # Generator gradient

function initweights(hidden,input, output)
    𝗪 = Any[];
    x = input
    for h in [hidden... output]
        push!(𝗪, atype(xavier(h,x)), atype(zeros(h, 1))) #FC Layers weights and bias
        x = h
    end
    return 𝗪  #return model params
end

function generate_and_save(𝗪,number,𝞗;fldr="generations/")
    Gz = G(𝗪[1],𝒩(𝞗[:ginp],number)) .> 0.5
    Gz = permutedims(reshape(Gz,(28,28,number)),(2,1,3))
    [save(fldr*string(i)*".png",Gray.(Gz[:,:,i])) for i=1:number]
end

#(if) train ? it updates model parameters : (else) it print losses
function runmodel(𝗪, data, 𝞗;dtst=nothing,optim=nothing,train=false,saveinterval=10)
    gloss=dloss=counter=0.0;
    B = 𝞗[:batchsize]
    for i=1:(train ? 𝞗[:epochs]:1)
        for (x,_) in data
            counter+=2B
            Gz = G(𝗪[1],𝒩(𝞗[:ginp],B)) #Fake Images
            train ? update!(𝗪[2], ∇d(𝗪[2],x,Gz), optim[2])      : (dloss += 2B*𝑱d(𝗪[2],x,Gz)) #update discriminator          
            z=𝒩(𝞗[:ginp],2B) #Sample z from Noise
            train ? update!(𝗪[1], ∇g(𝗪[1], 𝗪[2], z), optim[1]) : (gloss += 2B*𝑱g(𝗪[1],𝗪[2],z)) #update generator               
        end
        train ? runmodel(𝗪,dtst,𝞗;train=false) : println((gloss/counter,dloss/counter)) #Print average losses in each epoch
        i % saveinterval == 0 && generate_and_save(𝗪,10,𝞗)  # save 10 images to generations folder
    end
end

function main()
    𝞗 = Dict(:batchsize=>32,:epochs=>75,:ginp=>256,:genh=>[512],:disch=>[512],:optim=>Adam,:lr=>0.002);
    xtrn,ytrn,xtst,ytst = mnist()
    global dtrn = minibatch(xtrn, ytrn, 𝞗[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, 𝞗[:batchsize]; xtype=atype)
    𝗪 = (𝗪g,𝗪d)   = initweights(𝞗[:genh], 𝞗[:ginp], 784), initweights(𝞗[:disch], 784, 1)
    𝚶 = (𝚶pg,𝚶pd)  = optimizers(𝗪g,𝞗[:optim];lr=𝞗[:lr]), optimizers(𝗪d,𝞗[:optim];lr=𝞗[:lr])
    runmodel(𝗪, dtst, 𝞗;optim=𝚶, train=false) # initial losses
    runmodel(𝗪, dtrn, 𝞗;optim=𝚶,train=true, dtst=dtst)  # training
    𝗪,𝚶,𝞗,(dtrn,dtst)    # return weights,optimizers,options and dataset
end

main() #enjoy!
