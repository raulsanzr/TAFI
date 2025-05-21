# devtools::install_github("caravagnalab/mobster")
library(mobster)
library(dplyr)

pcawg_files <- list.files(path ='../data/processed/PCAWG', full.names = TRUE)

donors <- c()
shapes <- c()
# iterate over samples in PCAWG

print_shape <- function(i, pcawg_files){
  sample <- read.csv(pcawg_files[[i]])
  sample = sample[sample['VAF']<1,]
  # donor <- strsplit(strsplit(file,"/")[[1]][5], '.bed.gz')[[1]][1]
  x = mobster::mobster_fit(sample)
  shape <- x$best$shape
  
  plot(x$best)
  print(pcawg_files[[i]])
  
}
print_shape(1, pcawg_files)

for(file in pcawg_files){
  sample <- read.csv(file)
  sample = sample[sample['VAF']<1,]
  donor <- strsplit(strsplit(file,"/")[[1]][5], '.bed.gz')[[1]][1]
  x = mobster::mobster_fit(sample)
  shape <- x$best$shape
  
  donors <- c(donors, donor)
  shapes <- c(shapes, shape)
}

donors_b <- donors
shapes_b <- shapes

# df <- data.frame(id = donors, shape = shapes)
# sum(is.na(df$shape))
# write.csv(df, 'mobster_result.csv')

df <- read.csv('mobster_result.csv')
df_clean <- df[!is.na(df$shape),]
# df_clean$shape <- df_clean$shape + 1
tafi <- read.csv('../results/all_PCAWG.csv')
comp <- merge(tafi, df_clean, by.x = 'donor', by.y='id', all.x = F)
comp$tafi <- ifelse(comp$score_EXP > comp$score_WF, 'WF', 'EXP')

comp$sos <- ifelse(comp$sos_EXP > comp$sos_WF, 'WF', 'EXP')

table(comp$sos, comp$tafi)

library(ggplot2)

comp$tafi <- factor(comp$tafi, levels = c("WF", "EXP"))

ggplot(comp, aes(x=tafi, y=shape, group=tafi, fill=tafi))+
  geom_boxplot(width=0.6)+
  geom_jitter(aes(alpha=0.2), width=0.2)+
  theme_bw()+
  xlab('TAFI model')+
  ylab('MOBSTER shape')+
  theme(legend.position = 'none')+
  ggtitle('MOBSTER vs. TAFI classification')

# mydf = df %>% as_tibble() %>%
#   rename(Generation = d, Identity = cloneid) %>%
#   mutate(Identity = ifelse(Identity == 1, 'Ancestral', 'Subclone')) %>%
#   group_by(Generation, Identity) %>%
#   summarise(Population = n()) 
# 
# Muller_df <- ggmuller::get_Muller_df(data.frame(Parent = 'Ancestral', Identity = 'Subclone'), mydf)
# 
# cowplot::plot_grid(
#   ggmuller::Muller_pop_plot(Muller_df) + 
#     mobster:::my_ggplot_theme() +
#     guides(fill = guide_legend('Clone')) +
#     labs(title = 'Population size (simulation)', subtitle = "Muller plot (ggmuller)") +
#     scale_fill_manual(values = alpha(brewer.pal(2, "Set1"), alpha = .7)) +
#     geom_vline(xintercept = 17, linetype = 'dashed', size = .3),
#   mobster::plot.dbpmm(x$best) + labs(caption = NULL)
# )