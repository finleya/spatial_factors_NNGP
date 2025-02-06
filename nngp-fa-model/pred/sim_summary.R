rm(list=ls())
library(coda)
library(readr)
library(Matrix)

n.0 <- 1000
n <- 1000
p <- 3
q <- 4
h <- 10

w0.samps <- as.matrix(read_tsv("chain-1-w0", col_names=FALSE))
lambda.w0.samps <- as.matrix(read_tsv("chain-1-lambdaW0", col_names=FALSE))
z0.samps <- as.matrix(read_tsv("chain-1-z0", col_names=FALSE))

lambda.true <- as.matrix(read.table("../sim_data/lambda"))
w0.true <- read.table("../sim_data/w.ho")[,1]
z0.true <- read.table("../sim_data/z.ho")[,1]

w0.mu <- apply(w0.samps, 1, mean)
plot(w0.mu, w0.true)

lambda.w0.true <- diag(n)%x%lambda.true %*% w0.true

lambda.w0.mu <- apply(lambda.w0.samps, 1, mean)
plot(lambda.w0.mu, lambda.w0.true)

z0.mu <- apply(z0.samps, 1, mean)

plot(z0.mu, z0.true)

plot(z0.samps[,1], z0.true)


## ## Some double checks of the c code.
## lambda.samps <- as.matrix(read.table("../chain-1-lambda-thinned"))

## s <- 2
## plot(w0.samps[,s], w0.true)

## lambda.s <- matrix(lambda.samps[,s], nrow = h)
## lambda.s

## plot(lambda.w0.samps[,s], diag(n.0)%x%lambda.s%*%w0.samps[,s])

## beta.samps <- as.matrix(read.table("../chain-1-beta-thinned"))
## beta.s <- beta.samps[,s]

## x0 <- t(as.matrix(read.table("../sim_data/X.ho"))) ## Recall, this is stacked by outcome (see README).

## beta.true <- as.matrix(read.table("../sim_data/beta"))

## xb.s <- rep(0, h*n.0)
## xb.true <- rep(0, h*n.0)

## for(i in 1:n.0){
##     x <- x0[(i-1)+seq(1,nrow(x0),by=n.0),] ## I really must change the x stacking to match z.
##     x <- t(bdiag(as.list(data.frame(t(x)))))
##     xb.s[((i-1)*h+1):((i-1)*h+h)] <- x %*% beta.s
##     xb.true[((i-1)*h+1):((i-1)*h+h)] <- x %*% beta.true
## }

## plot(xb.s,  xb.true)

## plot(z0.samps[,s], xb.s+diag(n.0)%x%lambda.s%*%w0.samps[,s]) ##Looks good. The difference is because the prediction has eps in included.
