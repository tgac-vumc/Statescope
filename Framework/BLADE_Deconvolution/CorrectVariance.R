#!/usr/bin/env Rscript
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CorrectVariance.R
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# Script to mix counts from two seperate datasets
#
# Author: Jurriaan Janssen (j.janssen4@amsterdamumc.nl)
#
# Usage:
# Rscript scripts/Mix_data.R --dataset1 Wu --dataset2 Kim -o Mix
#
# TODO:
# 1)
#
# History:
#  20-12-2024: File creation, write code
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 0.1  Import Libraries
if(!'scran' %in% installed.packages()){devtools::install_github("Danko-Lab/BayesPrism/BayesPrism")}
if(!'optparse' %in% installed.packages()){install_packages("optparse")}
suppressMessages(library(optparse))
suppressMessages(library(scran))

#-------------------------------------------------------------------------------
# 1.1 Parse command line arguments
#-------------------------------------------------------------------------------
option_list = list(
    make_option(c("-i", "--input_mean"), action="store", default=NA, type='character',
                help="input Means"),
    make_option(c("-i", "--input_var"), action="store", default=NA, type='character',
                help="input Variance"),
    make_option(c("-o", "--output"), action="store", default=NA, type='character',
                help="output corrected Variance"))
    
    args = parse_args(OptionParser(option_list=option_list))
#-------------------------------------------------------------------------------
# 2.1 Read data
#-------------------------------------------------------------------------------
# read data
Mean <- read.delim(args$input_mean)
Var <- read.delim(args$input_var)

#-------------------------------------------------------------------------------
# 2.3 Fit mean-variance trend to obtain new std's
#-------------------------------------------------------------------------------
New_std <- matrix(nrow=nrow(Mean),ncol=ncol(Mean))
for(i in seq(length(ncol(Mean)))){
    trend <- fitTrendVar(mean[,i],variance[,i])$trend
    New_std[,i] <- trend(mean[,i])
}


write.table(NewStd, args$output, quote = F, row.names=F,sep = '\t')
