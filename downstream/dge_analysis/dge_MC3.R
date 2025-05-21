# Required packages
library(stringr)
library(DESeq2)
library(clusterProfiler)
library(biomaRt)
library(org.Hs.eg.db)
library(BiocParallel)
library(ggbeeswarm)
library(pals)
# library(quantmod)
library(DEGreport)

# READ THE DATA

current_dir <- getwd()
setwd(paste0(current_dir,'/data'))

# counts of each gene for all the donors
counts <- read.csv('MC3_counts.tsv', header=T)

# additional information of the donors 
metadata <- read.csv('gdc_sample_sheet_all.csv', header=T)

# model <- 'CN7'
# cancertype <- 'SKCM'

# # predicted classification of each donor
# class_WF <- read.csv(paste0('Documents/MC3/groups/',model,'/WF_df1.csv'), header=T)
# class_WF$Class <- 'WF'
# class_CC <- read.csv(paste0('Documents/MC3/groups/',model,'/CC_df1.csv'), header=T)
# class_CC$Class <- 'CC'
# class_EXP <- read.csv(paste0('Documents/MC3/groups/',model,'/EXP_df1.csv'), header=T)
# class_EXP$Class <- 'EXP'

scores <- read.csv('../../../results/all_MC3.csv')

class <- scores[c('donor', 'score_WF', 'score_EXP')]
class$mode <- ifelse(class$score_WF > class$score_EXP, "WF", "EXP")
class$sampleId <- class$donor

#------------------------------------------------------------------------------#

# FORMAT THE DATA

# change donors name
colnames(counts) <- str_replace(colnames(counts),'X','')
colnames(counts) <- str_replace(colnames(counts),'[.]','-')
colnames(counts) <- str_replace(colnames(counts),'[.]','-')
colnames(counts) <- str_replace(colnames(counts),'[.]','-')
colnames(counts) <- str_replace(colnames(counts),'[.]','-')

class[,c(1,4)]

classification <- class[,c(1,4)] # rbind(class_WF[,c(1,6)], class_CC[,c(1,7)],class_EXP[,c(1,6)])
colnames(classification) <- c('Sample.ID', 'Class')

# keep the first 15 characters of the IDs
classification$Sample.ID <- str_extract(classification$Sample.ID, "^.{15}")
metadata$Sample.ID <- str_extract(metadata$Sample.ID, "^.{15}")

#classification$Sample.ID <- classification$name
metadata <- metadata[(table(metadata$Sample.ID) == 1),] # remove duplicated samples

# add the classification to the metadata
metadata <- merge(classification, metadata, by='Sample.ID')
metadata$cancerType <- str_replace_all(string = metadata$Project.ID, replacement = '', pattern = 'TCGA-')

# filter by cancer type
metadata <- metadata[metadata$cancerType==cancertype,] 

metadata$Class <- as.factor(metadata$Class)

# reorder the counts data frame to match it with the metadata
counts_new <- counts[-c(1:5), c('gene', metadata$File.Name)]  # keep the counts present in the metadata filtered
colnames(counts_new) <- c('gene', metadata$Sample.ID)
rownames(metadata) <- metadata$Sample.ID
  
# check that they are correctly sorted (TRUE)
all(rownames(metadata) %in% colnames(counts_new))
all(rownames(metadata) == colnames(counts_new[2:length(counts_new)]))

#..............................................................................#
tab <- table(metadata$cancerType, metadata$Class)
prop.table(tab, margin = 2)

#..............................................................................#

#------------------------------------------------------------------------------#

# QUALITY CONTROL

# Principal component analysis

rld <- vst(dds, blind=T) # run dds below

plotPCA(rld, intgroup="Class")+
  ggtitle('PCA by tumor growth model (MC3)')+
  theme_bw()+
  labs(colour="Model")
  
resCov <- degCovariates(log2(counts(dds)+0.5),
                        colData(dds))

counts <- counts(dds, normalized = TRUE)
design <- as.data.frame(colData(dds))

degCheckFactors(counts[, 1:50])

degQC(counts, metadata$Class, pvalue = res[["pvalue"]])
#------------------------------------------------------------------------------#

# DIFFERENTIAL EXPRESSION ANALYSIS

dds <- DESeqDataSetFromMatrix(countData = counts_new, 
                              colData = metadata,
                              design = ~Class, 
                              tidy = TRUE)

# parallelization
register(MulticoreParam(6)) # using 6 CPUs to speed up
gc()

dds <- DESeq(dds, parallel=T)
res <- results(dds, contrast=c("Class","WF","EXP"))
res <- as.data.frame(res[order(res$padj),]) # sort the genes by p-value
head(res, 10)

# save the results
# write.csv(res, paste0('Documents/MC3/after_liftover/',model,'/res_STAD.csv'), row.names = T)

#------------------------------------------------------------------------------#

# GENE SET ENRICHMENT ANALYSIS

# significant genes below a p-value threshold
sig_genes <- res[(res$padj < 0.01 & is.na(res$padj)==F) ,]

# separating genes in downregulated (WF < EXP) and upregulated (WF > EXP) 
upreg_genes <- sig_genes[sig_genes$log2FoldChange>0,]
# write.table(gsub("\\..*","",rownames(upreg_genes)), 'Documents/GSEA/results/SKCM_up.txt', row.names = F, col.names = F, quote = F)
downreg_genes <- sig_genes[sig_genes$log2FoldChange<0,]
# write.table(gsub("\\..*","",rownames(downreg_genes)), 'Documents/GSEA/results/SKCM_down.txt', row.names = F, col.names = F, quote = F)

# converting ENSEMBL gene IDs into ENTREZ ones (needed for the GSEA)
mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
genes_entrez <- getBM(filters = "ensembl_gene_id",
                      attributes = c("ensembl_gene_id","entrezgene_id"),
                      values = unique(gsub("\\..*","",rownames(upreg_genes))), mart = mart)

## Gene Ontology: Biological Process (BP)
GO_BP <- enrichGO(gene = genes_entrez$entrezgene_id, OrgDb = org.Hs.eg.db,
                  ont = "BP")
head(GO_BP@result[,c(1,2,3,7)], 10) # Top 10
# write.csv(GO_BP@result, paste0('Documents/MC3/groups/GO_BP_up_STADbyEye.csv'), row.names = F) 

## KEGG Pathways
KEGG <- enrichKEGG(gene = genes_entrez$entrezgene_id, organism = 'hsa', keyType = 'kegg', pAdjustMethod = "fdr")
head(KEGG@result[,c(1,2,3,6)], 10) # Top 10
# write.csv(KEGG@result, paste0('Documents/MC3/groups/KEGG_up_STAD_byEye.csv'), row.names = F) 

#------------------------------------------------------------------------------#

# ASSESSING RANDOM GROUPS PERFORMANCE

sizeWF <- nrow(metadata[metadata$Class=='WF',]) # number of WF samples
sizeEXP <- nrow(metadata[metadata$Class=='EXP',]) # number of EXP samples

nsig_list <- data.frame('sig'=nrow(sig_genes), 'Grouping'='clas') # create a df to store the results with the actual number of sig. genes.

for (i in 1:20){ # 20 iterations
  metadata_new <- metadata
  metadata_new$Class <- 'EXP' # set all as EXP 
  metadata_new[sample(nrow(metadata_new), sizeWF), 'Class'] <- 'WF' # set random sizeWF as WF
  dds <- DESeqDataSetFromMatrix(countData = counts_new,
                                colData = metadata_new,
                                design = ~Class, tidy = TRUE) # DEA
  dds <- DESeq(dds, parallel=T)
  res <- results(dds, contrast=c("Class","WF","EXP"))
  res <- as.data.frame(res[order(res$padj),]) # sort the genes by p-value
  sig_genes_t <- res[(res$padj < 0.01 & is.na(res$padj)==F) ,]
  df_new <- data.frame('sig'=nrow(sig_genes_t), 'Grouping'='random') # save the total number of significant genes
  nsig_list <- rbind(nsig_list, df_new)
}

# plot the performance of finding significant genes of the random groups compared to the real TAFI classification
ggplot(nsig_list, aes(x=Grouping, y=sig, colour=Grouping))+
  geom_point()+
  theme_bw()+
  theme(legend.position = 'None')+
  ylab('number of significant genes')+
  ggtitle('Comparing the grouping of the classifier against 20 random (STAD)')

# -------------------------

library(stringr)
library(matrixStats)
library(Rtsne)

# Convert Uniprot ids into Ensembl ids
neg_pro <- read.csv('../HMF/neg_pro_ens.tsv', header=T,sep='\t')
neg_pro_ids <-  str_replace(neg_pro$To, "\\..*", "")
pos_pro <- read.csv('../HMF/pos_pro_ens.tsv', header=T,sep='\t')
pos_pro_ids <- str_replace(pos_pro$To, "\\..*", "")

counts_new$gene <- str_replace(counts_new$gene, "\\..*", "")
rownames(counts_new) <- counts_new$gene
counts_short <- na.omit(counts_new[c(pos_pro_ids,neg_pro_ids),])

# df. col= genes, rows = ctype
metadata <- metadata[!duplicated(metadata$File.Name),]
tissues <- sort(unique(metadata$cancerType))
tissues <- tissues[tissues != c('Unknown')]
newdf <- data.frame()

for(i in 1:length(tissues)){
  names_tissue <- metadata$File.Name[metadata$cancerType==tissues[i]]
  indx <- match(names_tissue,colnames(counts_short))
  indx <- as.numeric(na.omit(indx))
  if (length(indx) > 1){
    counts_ctype <- counts_short[rownames(counts_short), indx]
    means <- rowMeans(counts_ctype)
    newdf <- rbind(newdf, means)
    rownames(newdf)[nrow(newdf)] <- tissues[i]
  }
}

colnames(newdf) <- rownames(counts_ctype)

### pca
# Load necessary libraries

pal.bands(alphabet, alphabet2, cols25, glasbey, kelly, polychrome,
          stepped, tol, watlington,
          show.names=FALSE)

# Step 1: Perform PCA
# We transpose the data frame because `prcomp` expects variables (genes) in columns and observations (tissues) in rows
pca_result <- prcomp(newdf)
pca_data <- as.data.frame(pca_result$x)
pca_data$tissue <- rownames(pca_data)
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Step 3: Plot the PCA
# Plot the first two principal components (PC1 and PC2) and color by tissue
ggplot(pca_data, aes(x = PC1, y = PC2, color = tissue)) +
  geom_point(size = 3) +  # Size of points
  scale_color_manual(values=as.vector(polychrome(33))) +
  labs(title = "PCA of the mean expression of proliferation genes per cancer types",
       x = paste0("Principal Component 1 (", round(explained_variance[1] * 100, 1), "% variance)"),
       y = paste0("Principal Component 2 (", round(explained_variance[2] * 100, 1), "% variance)")) +
  theme_minimal() +
  theme(legend.position = "right")

# k-means
k <- 3
kmeans_result <- kmeans(newdf, centers = k, nstart = 25)
as.data.frame(kmeans_result$cluster)
