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
library(edgeR)
library(ggfortify)
library(EnsDb.Hsapiens.v79)
library(DEGreport)
library(ggplot2)
library(pals)
library(zoo)
library(ggrepel)

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

scores <- read.csv('../../../results/all_HMF.csv')
class <- scores[c('donor', 'score_WF', 'score_EXP')]
class$mode <- ifelse(class$score_WF > class$score_EXP, "WF", "EXP")
class$sampleId <- class$donor


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

metadata_HMF <- metadata_HMF[c('sampleId', 'type', 'mode', 'primaryTumorSubType', 'consolidatedTreatmentType')]

# filter by cancer type
# metadata_HMF <- metadata_HMF[metadata_HMF$type=='SKCM',]
# metadata_HMF <- metadata_HMF[!(metadata_HMF$type %in% c('Adrenal gland', 'DLBC', 'HNSC', 'LAML', 'STAD', 'THCA', 'Unknown')), ]

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

#..............................................................................#
tab <- table(metadata_HMF$type, metadata_HMF$mode)
prop.table(tab, margin = 1)
#..............................................................................#

#------------------------------------------------------------------------------#

# DIFFERENTIAL EXPRESSION ANALYSIS

dds <- DESeqDataSetFromMatrix(countData = counts_predicted,
                              colData = metadata_HMF,
                              design = ~ mode, tidy = TRUE)

library(Polychrome)
palette23 <- as.vector(polychrome(23))
# Principal component analysis
rld <- vst(dds, blind=T)
plotPCA(rld, intgroup = "mode") +
  ggtitle("PCA by tumor type") +
  # scale_color_manual(values = palette23) +
  theme_bw() +
  theme(legend.title = element_text(size = 10))

register(MulticoreParam(4)) # using multiple CPUs to speed up DESeq
gc()

dds <- DESeq(dds, parallel=T)
res <- results(dds, contrast=c("mode",'WF','EXP'))
res <- as.data.frame(res[order(res$padj),]) # sort the genes by p-value
head(res, 10)

# # save the results
write.csv(res, paste0('res_by_mode_type.csv'), row.names = T)

res <- read.csv('data/res_HMF_modeANDtype_12052025.csv')

# genes with a adjusted pvalue lower than 0.01
sig_genes <- res[(res$padj < 0.05 & is.na(res$padj)==F) ,]

upreg_genes <- res[res$padj < 0.05 & res$log2FoldChange > 1,] # WF > EXP
downreg_genes <- res[res$padj < 0.05 & res$log2FoldChange < -1,] # EXP > WF

res$significance <- "Not Significant"
res$significance[res$padj < 0.05 & res$log2FoldChange > 1] <- "Upregulated: WF>EXP"
res$significance[res$padj < 0.05 & res$log2FoldChange < -1] <- "Downregulated: WF<EXP"

# Replace NA values in padj to avoid plotting issues
res$padj[is.na(res$padj)] <- 1

# Volcano plot

top_genes <- res[(res$padj < 0.05 & abs(res$log2FoldChange) > 2) | res$padj < 10**-10, ]

top_genes2 <- getBM(filters = "ensembl_gene_id",
                   attributes = c("ensembl_gene_id","hgnc_symbol"),
                   values = top_genes$X, mart = mart)
top_genes$X <- top_genes2$hgnc_symbol


ggplot(res, aes(x = log2FoldChange, y = -log10(padj), color = significance)) +
  geom_point(alpha = 0.8, size = 1.5) +
  geom_text_repel(data = top_genes, 
                  aes(label = X),  # X should be your gene name column
                  size = 3,
                  max.overlaps = 20) +
  scale_color_manual(values = c("Upregulated: WF>EXP" = "red", 
                                "Downregulated: WF<EXP" = "blue", 
                                "Not Significant" = "grey")) +
  theme_minimal() +
  labs(x = "Log2 Fold Change",
       y = "-Log10 Adjusted P-value",
       color = "")

#------------------------------------------------------------------------------#

# GENE SET ENRICHMENT

mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")

## Converting ENSEMBL gene IDs into ENTREZ ones (needed for the GSEA)
genes_entrez <- getBM(filters = "ensembl_gene_id",
                      attributes = c("ensembl_gene_id","entrezgene_id"),
                      values = unique(downreg_genes$X), mart = mart)


## Gene Ontology: Biological Process (BP)
GO_BP <- enrichGO(gene = genes_entrez$entrezgene_id, OrgDb = org.Hs.eg.db,
                  ont = "BP") # biological process

a <- GO_BP@result[,c(2,3,9)]
a[a$p.adjust < 0.05,]
# head(GO_BP@result[,c(2,3,7)], 10) # Top 10
# write.csv(GO_BP@result, paste0('Documents/HMF/DE/GOBP_down_lung.csv'), row.names = F)

dotplot(GO_BP, showCategory = 10) + 
  ggtitle("GO: Biological Process - Downregulated genes (EXP>WF)")

## KEGG Pathways
KEGG <- enrichKEGG(gene = genes_entrez$entrezgene_id, organism = 'hsa', keyType = 'kegg', pAdjustMethod = "fdr")
# head(KEGG@result[,c(4,5,9)], 10) # Top 10
# write.csv(KEGG@result, paste0('Documents/HMF/DE/KEGG_down_lung.csv'), row.names = F)

KEGG@result[,c(4,5,11)]

dotplot(KEGG, showCategory = 10) + 
  ggtitle("KEGG Pathways - Downregulated genes (EXP>WF)")

# -----------------------------------

tab <- table(metadata_HMF$consolidatedTreatmentType, metadata_HMF$mode)
prop.table(tab, margin = 2)

# ---------------------------------

a <- read.csv('../../prop.csv')

ggplot(a, aes(type, EXP_PCAWG))+
  geom_jitter()+
  geom_jitter(aes(type, EXP_MC3))+
  geom_jitter(aes(type, WF_PCAWG))+
  geom_jitter(aes(type, WF_MC3))
