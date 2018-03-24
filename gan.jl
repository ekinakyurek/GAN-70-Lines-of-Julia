using Knet,ArgParse,Images
include(Pkg.dir("Knet","data","mnist.jl"))
global atype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}

function main()
    𝞗 = Dict(:batchsize=>32,:epochs=>100,:ginp=>256,:genh=>[512],:disch=>[512],:optim=>"Adam(lr=0.0002)");
    xtrn,ytrn,xtst,ytst = mnist()
    global dtrn = minibatch(xtrn, ytrn, 𝞗[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, 𝞗[:batchsize]; xtype=atype)
    𝗪  = (𝗪g,𝗪d)    = initweights(𝞗[:genh], 𝞗[:ginp], 784), initweights(𝞗[:disch], 784, 1)
    𝚶   = (𝚶pg,𝚶pd) =  initoptim(𝗪g,𝞗[:optim]), initoptim(𝗪d,𝞗[:optim])
    runmodel(𝗪, dtst, 𝞗;optim=𝚶, train=false) # initial losses
    runmodel(𝗪, dtrn, 𝞗;optim=𝚶,train=true, dtst=dtst)  # training
    return 𝗪,𝚶,dtrn,dtst,𝞗    # return weights,optimizers,dataset,options
end

#Generate and Save
function generate_and_save(𝗪,number,𝞗;fldr="generations/")
    Gz = G(𝗪[1],𝒩(𝞗[:ginp],number)) .> 0.5
    Gz = permutedims(reshape(Gz,(28,28,number)),(2,1,3))
    [save(fldr*string(i)*".png",convert(Array{Gray{N0f8},2},Gz[:,:,i])) for i=1:number]
end

#Train
function runmodel(𝗪, data, 𝞗;dtst=nothing,optim=nothing,train=false,saveinterval=10)
    gloss=dloss=total=0.0;
    B = 𝞗[:batchsize]
    for i=1:(train ? 𝞗[:epochs]:1)
        for (x,_) in data
            Gz = G(𝗪[1],𝒩(𝞗[:ginp],B)) #Generate Fake Images
            if train; update!(𝗪[2], ∇d(𝗪[2],x,Gz), optim[2]) #if train update discriminator
            else;     dloss += 2B*Dloss(𝗪[2],x,Gz); end      #else calculate loss
            z=𝒩(𝞗[:ginp],2B) #Sample z from Noise
            if train; update!(𝗪[1], ∇g(𝗪[1], 𝗪[2], z), optim[1]) #if train update generator
            else;    gloss += 2B*Gloss(𝗪[1],𝗪[2],z); end         #else calculate loss
            total+=2B
        end
        train ? runmodel(𝗪,dtst,𝞗;train=false):println((gloss/total,dloss/total)) #Print average losses in each epoch
        i % saveinterval == 0 && generate_and_save(𝗪,10,𝞗)  # save 10 images
    end
end

#Regular  MLP
function  mlp(w,x;p=0.0,activation=leakyrelu,outputactivation=sigm)
    for i=1:2:length(w)
        x = w[i]*dropout(mat(x),p) .+ w[i+1]  #FC Layer
        i<length(w)-1 && (x = activation.(x)) #Activation
    end
    outputactivation.(x) #Output
end

#Discriminator and Generators
D(w,x;p=0.0) = mlp(w,x;p=p)                                   #Discriminator
𝒩(input, batch) = convert(atype,randn(Float32, input, batch)) #SampleNoise
G(w,z;p=0.0) = mlp(w,z;p=p)                                   #Generator

#Initialize Weights
function initweights(hidden,input, output)
    𝗪 = Any[];
    x = input
    for h in [hidden... output]
        push!(𝗪, convert(atype, xavier(h,x)), convert(atype, zeros(h, 1))) #FC Layers weights and bias
        x = h
    end
    return 𝗪  #return model params
end

#Loss Functions and Gradients
global 𝜀=1e-8
Dloss(𝗪,x,Gz) = -mean(log.(D(𝗪,x)+𝜀)+log.(1-D(𝗪,Gz)+𝜀))/2 #Discriminator Loss
∇d = grad(Dloss) #Gradient according to discriminator loss
Gloss(𝗪g, 𝗪d, z) = -mean(log.(𝜀+D(𝗪d,G(𝗪g,z)))) #Generator Loss
∇g  = grad(Gloss) #Gradient according to generator loss

#Extensions
leakyrelu(x;α=0.2) = max(0,x) + α*min(0,x)                    #LeakyRelu activation
initoptim{T<:Number}(::KnetArray{T},otype)=eval(parse(otype)) #Optimizer initializations for KnetArray
initoptim{T<:Number}(::Array{T},otype)=eval(parse(otype))     #Optimizer initializations for Array
initoptim(a,otype)=map(x->initoptim(x,otype), a)
main() #RUN
