rm(list=ls())
library(coda)
library(readr)

n <- 1000
p <- 3
q <- 4
h <- 10

beta.samps <- t(read_tsv("chain-1-beta", col_names=FALSE))

phi.samps <- t(read_tsv("chain-1-phi", col_names=FALSE))

w.samps <- read_tsv("chain-1-w", col_names=FALSE)

Zw.samps <- read_tsv("chain-1-Zw", col_names=FALSE)

lambda.samps <- t(read_tsv("chain-1-lambda", col_names=FALSE))

fit.samps <- read_tsv("chain-1-fitted", col_names=FALSE)
z.hat <- apply(fit.samps, 1, mean)

z.true <- read.table("sim_data/z")[,1]

#for(i in 1:n){
#    plot(1:h, z.true[((i-1)*h+1):((i-1)*h+h)])
#    lines(1:h, z.hat[((i-1)*h+1):((i-1)*h+h)], col="blue")
#    readline(prompt = "Pause. Press <Enter> to continue...")
#}

w.true <- read.table("sim_data/w")[,1]
Zw.true <- read.table("sim_data/Zw")[,1]
beta.true <- read.table("sim_data/beta")[,1]
lambda.true <- as.vector(as.matrix(read.table("sim_data/lambda")))
phi.true <- as.vector(as.matrix(read.table("sim_data/phi")))

quants <- function(x){quantile(x, prob=c(0.5, 0.025, 0.975))}

q.plot <- function(x, y, main=""){
    cover.per <- round(sum(x > y[2,] & x < y[3,])/length(x)*100,2)
    plot(x, y[1,], ylim=range(c(x, y)), main=paste(main, " (cover %: ", cover.per, ")", sep=""), xlab="True", ylab="Estimated")
    lines(seq(min(y), max(y), length.out=2), seq(min(y), max(y), length.out=2), col="red")
    arrows(x, y[1,], x, col="blue", y[2,], length=0.02, angle=90)
    arrows(x, y[1,], x, col="blue", y[3,], length=0.02, angle=90)
}

z.q <- apply(fit.samps, 1, quants)
w.q <- apply(w.samps, 1, quants)
Zw.q <- apply(Zw.samps, 1, quants)
beta.q <- apply(beta.samps, 2, quants)
lambda.q <- apply(lambda.samps, 2, quants)
phi.q <- apply(phi.samps, 2, quants)

#png(file="params.png")
par(mfrow=c(2, 3))
q.plot(w.true, w.q, "w")
q.plot(Zw.true, Zw.q, "Zw")
#q.plot(z.true, z.q, "z")
q.plot(beta.true, beta.q, "beta")
q.plot(lambda.true[-which(lambda.true == 1 | lambda.true == 0)], lambda.q[,-which(lambda.true == 1 | lambda.true == 0)], "lambda")
q.plot(phi.true, phi.q, "phi")
#dev.off()
