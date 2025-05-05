# Required packages
library(stringr)
library(DESeq2)
library(clusterProfiler)
library(biomaRt)
library(org.Hs.eg.db)
library(BiocParallel)
library(ggbeeswarm)
library(pals)
library(quantmod)
library(edgeR)
library(ggfortify)
library(EnsDb.Hsapiens.v79)
library(DEGreport)
library(ggplot2)
library(pals)

#------------------------------------------------------------------------------#

# READ THE DATA

current_dir <- getwd()
setwd(paste0(current_dir,'/data'))

counts <- read.csv('all_total_counts.csv', header=T)
# row.names(counts) <- counts$geneID
counts <- counts[,-c(1)]

# filter low expressed genes
keep <- rowSums(edgeR::cpm(counts[,2:ncol(counts)])>1) >= 2

## file containing additional information of the donors 
metadata <- read.csv('metadata_update.tsv', header=T, sep='\t')

scores <- read.csv('predictions/ABC_scores.csv')
class <- scores[c('donor', 'ngauss_gauss_clas_strict')]

class$cons <- ifelse(class$ngauss_gauss_clas_strict == 0.5, NA, class$ngauss_gauss_clas_strict) # Remove class = 0.5
class <- na.omit(class)
class <- class[c('donor', 'cons')]
colnames(class)[1] <- 'sampleId'
class$cons <- as.factor(class$cons)

#------------------------------------------------------------------------------#

# FORMAT THE DATA

metadata_HMF <- merge(metadata, class, by='sampleId')

# Build type short
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Breast'] <- 'BRCA'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Ovary'] <- 'OV'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Pancreas'] <- 'PAAD'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Mesothelium'] <- 'MESO'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Liver'] <- 'LIHC'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Thyroid gland'] <- 'THCA'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Head and neck'] <- 'HNSC'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Skin'] <- 'SKCM'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Stomach'] <- 'STAD'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Bile duct'] <- 'CHOL'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Urothelial tract'] <- 'BLCA'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Esophagus'] <- 'ESCA'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Thymus'] <- 'ESCA'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Prostate'] <- 'PRAD'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Bone marrow'] <- 'LAML'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Hepatobiliary system'] <- 'LIHC'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Testis'] <- 'TGCT'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Lymphoid tissue'] <- 'DLBC'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Bone/Soft tissue'] <- 'SARC'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Colorectum'] <- 'Colorectum'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Lung'] <- 'Lung'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Kidney'] <- 'Kidney'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Uterus'] <- 'Uterus'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Adrenal gland'] <- 'Adrenal gland'
metadata_HMF$type[metadata_HMF$primaryTumorLocation == 'Nervous system'] <- 'Nervous system'
metadata_HMF$type <- na.fill(metadata_HMF$type, 'Unknown')

row.names(metadata_HMF) <- metadata_HMF$sampleId

metadata_HMF <- metadata_HMF[c('sampleId', 'type', 'cons', 'primaryTumorSubType', 'consolidatedTreatmentType')]

# filter by cancer type
metadata_HMF <- metadata_HMF[metadata_HMF$type=='Lung',]

# keep the patients with predictions
HMF_patients <- intersect(metadata_HMF$sampleId, colnames(counts))

# reorder the counts
counts_predicted <- counts[keep,c('geneID',HMF_patients)]
metadata_HMF <- metadata_HMF[!duplicated(metadata_HMF$sampleId),]
row.names(metadata_HMF) <- metadata_HMF$sampleId

# reorder the metadata
metadata_HMF <- metadata_HMF[HMF_patients,]

# check that they are correctly sorted (both sould return TRUE)
all(rownames(metadata_HMF) %in% colnames(counts_predicted))
all(rownames(metadata_HMF) == colnames(counts_predicted[2:length(counts_predicted)]))

# HMF suggest to remove the following genes from the analysis https://github.com/hartwigmedical/hmftools/tree/master/isofox
rows_to_remove <- c('ENSG00000274012', #RN7SL2
                    'ENSG00000276168', #RN7SL1
                    'ENSG00000278771', #RN7SL3
                    'ENSG00000263740', #RN7SL4P
                    'ENSG00000265735', #RN7SL5P
                    'ENSG00000283293') #RN7SK

counts_predicted <- counts_predicted[!(counts_predicted$geneID %in% rows_to_remove), ]
metadata_HMF$cons <- ifelse(metadata_HMF$cons == 0, 'WF', 'EXP')

#------------------------------------------------------------------------------#

# DIFFERENTIAL EXPRESSION ANALYSIS

dds <- DESeqDataSetFromMatrix(countData = counts_predicted,
                              colData = metadata_HMF,
                              design = ~cons, tidy = TRUE)

# Principal component analysis
rld <- vst(dds, blind=T)
plotPCA(rld, intgroup="cons")+
  ggtitle('PCA by class (HMF)')+
  # scale_color_manual(values=as.vector(polychrome(24))) +
  theme_bw()+
  labs(colour="Class")

register(MulticoreParam(4)) # using multiple CPUs to speed up DESeq
gc()

dds <- DESeq(dds, parallel=T)
res <- results(dds, contrast=c("cons",'WF','EXP'))
res <- as.data.frame(res[order(res$padj),]) # sort the genes by p-value
head(res, 10)

# # save the results
# write.csv(res, paste0('Documents/HMF/DE/res_lung.csv'), row.names = T)

# genes with a adjusted pvalue lower than 0.01
sig_genes <- res[(res$padj < 0.01 & is.na(res$padj)==F) ,]

upreg_genes <- sig_genes[sig_genes$log2FoldChange>0,] # WF > EXP
downreg_genes <- sig_genes[sig_genes$log2FoldChange<0,] # EXP > WF

#------------------------------------------------------------------------------#

# GENE SET ENRICHMENT

mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

## Converting ENSEMBL gene IDs into ENTREZ ones (needed for the GSEA)
genes_entrez <- getBM(filters = "ensembl_gene_id",
                      attributes = c("ensembl_gene_id","entrezgene_id"),
                      values = unique(rownames(downreg_genes)), mart = mart)

## Gene Ontology: Biological Process (BP)
GO_BP <- enrichGO(gene = genes_entrez$entrezgene_id, OrgDb = org.Hs.eg.db,
                  ont = "BP") # biological process

# head(GO_BP@result[,c(2,3,7)], 10) # Top 10
# write.csv(GO_BP@result, paste0('Documents/HMF/DE/GOBP_down_lung.csv'), row.names = F)

dotplot(GO_BP, showCategory = 20) + 
  ggtitle("GO: Biological Process (EXP > WF)")

## KEGG Pathways
KEGG <- enrichKEGG(gene = genes_entrez$entrezgene_id, organism = 'hsa', keyType = 'kegg', pAdjustMethod = "fdr")
# head(KEGG@result[,c(4,5,9)], 10) # Top 10
# write.csv(KEGG@result, paste0('Documents/HMF/DE/KEGG_down_lung.csv'), row.names = F)

dotplot(KEGG, showCategory = 25) + 
  ggtitle("KEGG Pathways (EXP > WF)")
