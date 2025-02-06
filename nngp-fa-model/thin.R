rm(list=ls())

beta.s <- read.table("chain-1-beta")
ncol(beta.s)

##Take a sub sample if you want.
sub <- sample.int(ncol(beta.s), 10)
length(sub)
sub <- paste0(sub,collapse=",")

system(paste0("cut -f", sub," chain-1-beta > chain-1-beta-thinned"))
system(paste0("cut -f", sub," chain-1-phi > chain-1-phi-thinned"))
system(paste0("cut -f", sub," chain-1-lambda > chain-1-lambda-thinned"))
system(paste0("cut -f", sub," chain-1-tauSq > chain-1-tauSq-thinned"))
system(paste0("cut -f", sub," chain-1-w > chain-1-w-thinned"))

