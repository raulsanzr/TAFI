# RNA-seq TPM's exploration: PCAWG data

library(ggplot2)
library(gplots)
library(ggfortify)
library(gridExtra)
library(limma)
library(pals)
library(dplyr)
library(propr)
library(stringr)
library(smd)

current_dir <- getwd()
setwd(paste0(current_dir,'/data'))

# Reading the data from https://www.ebi.ac.uk/gxa/experiments/E-MTAB-5423/Downloads
tpms_data <- as.data.frame(read.table(file='E-MTAB-5423-query-results.tpms.tsv', sep='\t', header=TRUE))
## 1350 donors
## 56717 genes

#..............................................................................#

# Applying a threshold | 23991 out of 56717 genes were removed |
## NA's removal: Into 0's
tpms_data[is.na(tpms_data)] <- 0

## Threshold: genes having TPM values higher or equal than 0.1 in at least 20% of the donors
tpms_threshold <- tpms_data[,3:ncol(tpms_data)] > 0.1
keep <- rowSums(tpms_threshold) > 0.2*(ncol(tpms_threshold))
tpms_flt <- tpms_data[keep,]

#------------------------------------------------------------------------------#

# Keeping just the target genes
target_genes <- read.csv(file = 'Documents/genelists/MC3_SKCM_down.csv', header=F)
tpms_target <- tpms_flt[tpms_flt$Gene.Name %in% target_genes$V1,]

# Creating the metadata
expanded_meta <- colnames(tpms_target[-c(1,2)])

## Tumour status
status <- sub('.*[0-9]','',expanded_meta)
### Renaming status labels
status <- gsub('...metastatic.tumour', 'metastatic_tumour', status)
status <- gsub('...normal', 'normal', status)
status <- gsub('...primary.tumour', 'primary_tumour', status)
status <- gsub('...recurrent.tumour', 'recurrent_tumour', status)

## Cancer type
type <- sub('[.].*','',expanded_meta)

### Renaming type labels
type <- gsub('B', 'B-cell', type)
type <- gsub('chromophobe', 'renal', type)
type <- gsub('cholangiocarcinoma', 'liver', type)
type <- gsub('cholangiocarcinoma', 'liver', type)
type <- gsub('chronic', 'leukemia', type)
type <- gsub('follicular', 'thyroid', type)
type <- gsub('invasive', 'lobular', type)

## Donor ID's
ID <- paste0('DO', as.numeric(gsub("\\D", "", expanded_meta)))

# Visual exploration: PCA
## Creating a numeric df
df <- as.data.frame(t(tpms_target[,c(-1,-2)]))
colnames(df) <- tpms_target$Gene.Name

## Adding the metadata
df$Type <- type
df$Status <- status
df$ID <- ID

# Differential proportionality approach (https://cran.microsoft.com/snapshot/2018-04-24/web/packages/propr/vignettes/e_differential.html)
## Setting up the data

# TAFI output files: contain the predicted model for each donor
class_WF <- read.csv(paste0('Documents/survival_analysis/predict_patients/WF_parameters_pred.csv'), header=T, sep='\t')
class_WF$Class <- 'WF'
class_CC <- read.csv(paste0('Documents/survival_analysis/predict_patients/CC_parameters_pred.csv'), header=T, sep='\t')
class_CC$Class <- 'CC'
class_EXP <- read.csv(paste0('Documents/survival_analysis/predict_patients/EXP_parameters_pred.csv'), header=T, sep='\t')
class_EXP$Class <- 'EXP'

classification <- rbind(class_WF[,c(1,11)], class_CC[,c(1,12)],class_EXP[,c(1,11)])
classification$name <- str_replace_all(string = classification$name, replacement = '', pattern = '.bed.gz')

# Merge metadata with prediction
colnames(classification)[1] <- 'ID'
classification_PCAWG <- classification[nchar(classification$ID) > 20, ] # keep the PCAWG donors
classification_PCAWG$ID <- sub("^.*_(.*?)_.*$", "\\1", classification_PCAWG$ID)

#------------------------------------------------------------------------------#

# merge the tpms with the classification
df_class <- merge(df, classification_PCAWG, by='ID')
df_class <- df_class[order(df_class$Class),]

#------------------------------------------------------------------------------#

## Principal Component Analysis
pca_res <- prcomp(df_class[,2:(ncol(df_class)-3)], scale=F)

### PC1 against PC2 coloured by tissue
autoplot(pca_res, data=df_class, colour='Status', x=1, y=2) +
  theme_bw() +
  ggtitle('') +
  scale_colour_manual(values=as.vector(polychrome(26)))

# # Function to generate a PCA per cancer type
# pca_bytype <- function(df){
#   pca_list <- list() # List to save the plots
#   for(i in levels(factor(df$Type))){
#     df_bytype <- df[df$Type %in% i,] # Iteration over the cancer types  
#     pca_res <- prcomp(df_bytype[,1:(ncol(df_bytype)-3)], scale=F) # pca
#     
#     pca_list[[i]] <- autoplot(pca_res, data=df_bytype, colour='Status', x=1, y=3) + 
#       theme_bw() + 
#       ggtitle(i)
#   }
#   return(pca_list)
# }
# 
# pca_list <- pca_bytype(df)
# grid.arrange(grobs=pca_list, nrow = 5, common.legend=TRUE)

#------------------------------------------------------------------------------#

# # Filter by cancer type
df_class <- df_class[df_class$Type == 'gastric',]

## Applying the propd method
pd <- propd(df_class[,2:(ncol(df_class)-3)], group=df_class$Class, alpha = NA, p = 100)

### Getting the p-values
pd <- updateF(pd)
results <- getResults(pd)

### Filtering results for comparisons involving housekeeping genes
housekeeping_genes <- c('GAPDH', 'TUBB')
results_flt <- results[results$Partner %in% housekeeping_genes | results$Pair %in% housekeeping_genes,]

# calculating the adjusted pvalues using the fdr method
results_flt$FDR <- p.adjust(results_flt$Pval, method="fdr")
# sort by ascending fdr
results_flt[order(results_flt$FDR),]

# List to save the index when a new cancer type starts in the df
# index_list <- list()
# for (i in levels(factor(df_class$type_short))){
#   for (row in 1:nrow(df_class)){
#     if(df_class[row,]$type_short == i){
#       index_list[[i]] <- row
#       break
#     }
#   }
# }

## Genes to compare 
gene1='FUT7' # proliferation-related gene
gene2='TUBB' # c('TUBB','GAPDH')

## Correlation plot between two genes
plot(pd@counts[, gene1], pd@counts[, gene2], col = ifelse(pd@group == "EXP","firebrick1", "dodgerblue"), xlab=gene1,
     ylab=gene2, main=paste0('Correlation plot between ', gene1, ' and ', gene2), pch=20)
legend('topright',c("WF","EXP"),cex=.8,col=c("dodgerblue","firebrick1"),pch=c(20))

## Ratio plot
plot(pd@counts[, gene1] / pd@counts[, gene2], col = ifelse(pd@group == "EXP","firebrick1", "dodgerblue"), xlab='Sample index',
     ylab=paste0(gene1,' / ',gene2), main=paste0('Ratio plot between ', gene1, ' and ', gene2), pch=20)
legend('topright',c("WF","EXP"),cex=.8,col=c("dodgerblue","firebrick1"),pch=c(20))
abline(a=mean(pd@counts[pd@group == "WF", gene1] / pd@counts[pd@group == "WF", gene2]), b=0, col='dodgerblue', lty=2)
abline(a=mean(pd@counts[pd@group == "EXP", gene1] / pd@counts[pd@group == "EXP", gene2]), b=0, col='firebrick1', lty=2)
abline(v=index_list, col='darkgray') # (when using all cancer types, to separate by type)
