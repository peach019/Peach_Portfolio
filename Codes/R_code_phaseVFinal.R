library(factoextra)
install.packages('NbClust')
library(NbClust)
library(cluster) 
library(gridExtra)
library(grid)
library(ggpubr)
install.packages('clusterCrit')
library(clusterCrit)

library(clustertend)

library(readr)
library(stats)
df_RFM2 <- read_csv("df_RFM2.csv")


df <- df_RFM2[3:5]
str(df)
ptm <- proc.time()
set.seed(123)
hopkins1 <- hopkins (df, n=nrow(df)-1)
hopkins1
proc.time() - ptm

dfs<-scale(df)
p<-as.data.frame(prcomp(df))
km.res <- eclust(dfs, "kmeans", k = 3, nstart = 25, graph = FALSE)

library(fpc)
# Statistics for k-means clustering
km_stats <- cluster.stats(dist(df), res.kmeans.4$cluster)
# Dun index
km_stats$dunn
# Corrected Rand index
clust_stats$corrected.rand
# VI
clust_stats$vi
km_stats.cl4<-cluster.stats(dist(df),res.kmeans.4$cluster)

clust_stats <- cluster.stats(d = dist(df),
                             species, km.res$cluster)


#Normalizing the dataset- Min-Max scaling
library(caret)
norm <- preProcess(df, method=c("range"))
df_norm <- predict(norm, df)
View(head(df_norm))


#PCA on normalized dataset
#pca plot
# prin_comp_n <- prcomp(n, scale = T)
# fviz_eig(prin_comp_n)

prin_comp <- prcomp(df_norm, scale = F) #since it is already scaled
fviz_eig(prin_comp)
prin_comp
names(prin_comp)
summary(prin_comp)
prin_comp$rotation
prin_comp$x[,1:2]

#same results by different method
library(stats)
prin_comp2 <- princomp(df_norm, scores = TRUE, cor = TRUE)
summary(prin_comp2)
prin_comp2$loadings

prin_comp$loadings
screeplot(prin_comp, main ="Scree Plot", xlab="Components")
screeplot(prin_comp, main="Scree Plot", type="line")


pr_var <- prin_comp$sdev^2
prop_varex <- pr_var/sum(pr_var)
prop_varex
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
#First 2 components extracted
data <- as.data.frame(prin_comp$x[,1:2])
View(head(data))

ggplot(data, aes(PC1, PC2)) +
  stat_ellipse(geom = "polygon", col = "black", alpha = 0.5) +
  geom_point(shape = 21, col = "black")

fviz_pca_ind(prcomp(df_norm), title = "PCA - Meter Data", palette = "jco", geom = "point", ggtheme = theme_classic(), legend = "bottom")

dim(data) #2980 2

#next steps
#Normalizing again before applying k means clustering as PCA data is negative too and this would endure adequate calculation for euclidean distance 

norm2 <- preProcess(data, method=c("range"))
data_norm <- predict(norm2, data)
View(head(data_norm))

#kmeans clustering using NBClust

nb <- NbClust(data_norm, distance = "euclidean", min.nc = 2,
              max.nc = 10, method = "kmeans")

fviz_nbclust(nb)


#kmeans clustering
set.seed(123)
km.res <- kmeans(data_norm, 3, nstart = 25)
#sizes 11, 2587, 382
print(km.res) #99.3%

library(factoextra)
fviz_nbclust(data_norm, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

#Visualizing k means clustering
fviz_cluster(km.res, data = data_norm,
             palette = c("#2E9FDF", "#FF0000", "#E7B800"),
             ellipse.type = "euclid", # Concentration ellipse
             star.plot = TRUE, # Add segments from centroids to items
             repel = TRUE, # Avoid label overplotting (slow)
             ggtheme = theme_minimal()
)

dim(data_norm)

k3<-kmeans(data_norm[100,],3,iter.max=100,nstart=50,algorithm="Lloyd")
s3<-plot(silhouette(k3$cluster,dist(data_norm,"euclidean")))

# Elbow method
fviz_nbclust(data_norm, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(data_norm, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

# Gap statistic
# nboot = 50 to keep the function speedy.
# recommended value: nboot= 500 for your analysis.
# Use verbose = FALSE to hide computing progression.
set.seed(123)
fviz_nbclust(data_norm, kmeans, nstart = 25, method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")



#PAM clustering
pam.res <- pam(data_norm,4)

fviz_cluster(pam.res,
             palette = c("#00AFBB", "#FC4E07", "#E7B800","#2E9FDF"), # color palette
             ellipse.type = "t", # Concentration ellipse
             repel = TRUE, # Avoid label overplotting (slow)
             ggtheme = theme_classic()
)

# K-means clustering
km.res.ec <- eclust(data_norm, "kmeans", k = 3, nstart = 25, graph = FALSE)





## no need to scale ### df.S <- scale(df , center = TRUE , scale = TRUE)

library(parallel)
detectCores()
# 8 Cores

###################### Check for percentage accurate partition
################### between_ss / total_SS

#### Time taken to run is also Calculated ############

############### For K=2 ########################################
################################################################

Rprof()
res.kmeans.2 <- kmeans(dfs,
                       centers = 2,
                       nstart = 100,
                       iter.max = 300)
print(res.kmeans.2)
Rprof(NULL)

summaryRprof()

############## OUTPUT ##################

# K-means clustering with 2 clusters of sizes 92095, 15411

# Cluster means:
#  Recency Frequency  Monetary
# 1 1.947967  5.314436  25.59688
# 2 1.482707 16.963403 113.62081

#Within cluster sum of squares by cluster:
#  [1] 26761465 28277807
# (between_SS / total_SS =  65.4 %)

######## Timing ##########

"$by.self
#self.time self.pct total.time total.pct
.Fortran                3.26    81.91       3.26     81.91
array                   0.36     9.05       0.38      9.55
asplit                  0.12     3.02       0.50     12.56
double                  0.06     1.51       0.06      1.51
duplicated.default      0.04     1.01       0.04      1.01
sample.int              0.04     1.01       0.04      1.01
do_one                  0.02     0.50       3.36     84.42
integer                 0.02     0.50       0.02      0.50
is.atomic               0.02     0.50       0.02      0.50
nrow                    0.02     0.50       0.02      0.50
print.default           0.02     0.50       0.02      0.50"

### $sampling.time
###[1] 3.98 secs


####### Plotting the clusters using PCA technique ########

# Dimension reduction using PCA
res.pca <- prcomp(df_RFM2[,-1],  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.kmeans.2$cluster)
# Add Species groups from the original data sett

# Data inspection
head(ind.coord)


# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "euclid",
  repel = TRUE,
  shape = "cluster", size = 1.5,  legend = "right", ggtheme = theme_minimal(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
# Cluster Graph Shows small amount of overlapping

##### Print the means of original data columns with clusters =2
dd <- cbind(df, cluster = res.kmeans.2$cluster)
temp        <- aggregate(dd, by=list(res.kmeans.2$cluster), FUN=mean)
temp2       <- t(temp)
temp2

"   [,1]       [,2]
Group.1    1.000000   2.000000
Recency    1.947967   1.482707
Frequency  5.314436  16.963403
Monetary  25.596881 113.620807
cluster    1.000000   2.000000"

" Cluster 1 = Customers with less monetary value and who are very less frequent
Cluster 2 = Customers with more monetary values or high purchases"

##### Changing the dimensionality to check the performance of K =2
### Remove the column Recency and check the quality of Kmeans

df1 = df[2:3]
str(df1)
Rprof()
res.kmeans.2_1 <- kmeans(df1,
                       centers = 2,
                       nstart = 100,
                       iter.max = 300)
print(res.kmeans.2)
Rprof(NULL)

summaryRprof()

############## OUTPUT ##################

"K-means clustering with 2 clusters of sizes 15414, 92092

Cluster means:
  Frequency  Monetary
1 16.962242 113.61224
2  5.314251  25.59545"

"Within cluster sum of squares by cluster:
[1] 28255328 26575254
 (between_SS / total_SS =  65.5 %)"

"$by.self
self.time self.pct total.time total.pct
.Fortran                2.58    78.18       2.58     78.18
array                   0.42    12.73       0.42     12.73
asplit                  0.12     3.64       0.54     16.36
aperm.default           0.04     1.21       0.04      1.21
duplicated.default      0.04     1.21       0.04      1.21
do_one                  0.02     0.61       2.62     79.39
do.call                 0.02     0.61       0.02      0.61
integer                 0.02     0.61       0.02      0.61
print.default           0.02     0.61       0.02      0.61
sample.int              0.02     0.61       0.02      0.61"

#$sample.interval
#[1] 0.02

#$sampling.time
#[1] 3.3

# Not much difference in terms of partition

####### Plotting the clusters using PCA technique ########

# Dimension reduction using PCA

res.pca <- prcomp(df_RFM2[,c(-1,-3)],  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.kmeans.2$cluster)
# Add Species groups from the original data sett

# Data inspection
head(ind.coord)


# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "euclid",
  repel = TRUE,
  shape = "cluster", size = 1.5,  legend = "right", ggtheme = theme_minimal(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
##### Print the means of original data columns with clusters = 2
dd <- cbind(df1, cluster = res.kmeans.2$cluster)
temp        <- aggregate(dd, by=list(res.kmeans.2$cluster), FUN=mean)
temp2       <- t(temp)
temp2

## In terms of graph(overlapping) and grouping there is not much difference#


#### Checking the Internal Indices on fist K = 2 #####


getCriteriaNames(TRUE)
intIdx <- intCriteria(as.matrix(df),res.kmeans.2$cluster,"Dunn")
length(intIdx)

### This is terminating the R not able to run may be because of length
# of Dataset. So please mention in report indices got aborted by R due to 
# length of dataset

####### End of Checking all Criteria with K=2 #########
########################################################




############### For K=3 ########################################
################################################################

Rprof()
res.kmeans.3 <- kmeans(df,
                       centers = 3,
                       nstart = 100,
                       iter.max = 300)
print(res.kmeans.3)
Rprof(NULL)

summaryRprof()

############## OUTPUT ##################

# K-means clustering with 3 clusters of sizes 75231, 6442, 25833

"Cluster means:
  Recency Frequency  Monetary
1 1.974744  4.438104  19.43355
2 1.314654 21.006830 152.80727
3 1.750358 10.902605  64.33496"

###################### Check for percentage accurate partition
################### between_ss / total_SS
"Within cluster sum of squares by cluster:
  [1] 9206293 9386481 9622960
(between_SS / total_SS =  82.3 %)"

######## Timing ##########

"$by.self
                     self.time self.pct total.time total.pct
.Fortran                6.42    84.25       6.42     84.25
array                   0.66     8.66       0.68      8.92
integer                 0.16     2.10       0.16      2.10
print.default           0.12     1.57       0.12      1.57
double                  0.10     1.31       0.10      1.31
asplit                  0.08     1.05       0.76      9.97
duplicated.default      0.04     0.52       0.04      0.52
is.atomic               0.02     0.26       0.02      0.26
sample.int              0.02     0.26       0.02      0.26"

#$sampling.time
#[1] 7.62 secs


####### Plotting the clusters using PCA technique ########

# Dimension reduction using PCA
res.pca <- prcomp(df_RFM2[,-1],  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.kmeans.3$cluster)
# Add Species groups from the original data sett

# Data inspection
head(ind.coord)


# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "euclid",
  repel = TRUE,
  shape = "cluster", size = 1.5,  legend = "right", ggtheme = theme_minimal(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
# Cluster Graph Shows small amount of overlapping

##### Print the means of original data columns with clusters =3
dd <- cbind(df, cluster = res.kmeans.3$cluster)
temp        <- aggregate(dd, by=list(res.kmeans.3$cluster), FUN=mean)
temp2       <- t(temp)
temp2

"      [,1]       [,2]      [,3]
Group.1    1.000000   2.000000  3.000000
Recency    1.974744   1.314654  1.750358
Frequency  4.438104  21.006830 10.902605
Monetary  19.433550 152.807268 64.334964
cluster    1.000000   2.000000  3.000000"

" Cluster 1 = Customers with less monetary value and who are very less frequent
Cluster 2 = Customers with more monetary values or high purchases
Cluster 3 = Average customers"

##### Changing the dimensionality to check the performance of K =3
### Remove the column Recency and check the quality of Kmeans

df1 = df[2:3]
str(df1)
Rprof()
res.kmeans.3_1 <- kmeans(df1,
                         centers = 3,
                         nstart = 100,
                         iter.max = 300)
print(res.kmeans.3_1)
Rprof(NULL)

summaryRprof()

############## OUTPUT ##################

"K-means clustering with 3 clusters of sizes 75224, 6442, 25840

Cluster means:
  Frequency  Monetary
1  4.437786  19.43146
2 21.006830 152.80727
3 10.901780  64.32887"

"Within cluster sum of squares by cluster:
[1] 9055160 9375285 9576912
 (between_SS / total_SS =  82.4 %)"

"$by.self
                     self.time self.pct total.time total.pct
.Fortran                5.12    84.21       5.12     84.21
array                   0.38     6.25       0.40      6.58
asplit                  0.22     3.62       0.62     10.20
integer                 0.12     1.97       0.12      1.97
print.default           0.10     1.64       0.10      1.64
duplicated.default      0.06     0.99       0.06      0.99
double                  0.04     0.66       0.04      0.66
do_one                  0.02     0.33       5.30     87.17
is.atomic               0.02     0.33       0.02      0.33"

# $sampling.time
# [1] 6.08

# Not much difference in terms of partition

####### Plotting the clusters using PCA technique ########

# Dimension reduction using PCA

res.pca <- prcomp(df_RFM2[,c(-1,-3)],  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.kmeans.3_1$cluster)
# Add Species groups from the original data sett

# Data inspection
head(ind.coord)


# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "euclid",
  repel = TRUE,
  shape = "cluster", size = 1.5,  legend = "right", ggtheme = theme_minimal(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
##### Print the means of original data columns with clusters = 3
dd <- cbind(df1, cluster = res.kmeans.3$cluster)
temp        <- aggregate(dd, by=list(res.kmeans.3$cluster), FUN=mean)
temp2       <- t(temp)
temp2

## In terms of graph(overlapping) and grouping there is not much difference#


#### Checking the Internal Indices on first K = 3 #####


getCriteriaNames(TRUE)
intIdx <- intCriteria(as.matrix(df),res.kmeans.2$cluster,"Dunn")
length(intIdx)

intIdx <- intCriteria(as.matrix(df),res.kmeans.2$cluster,"Silhouette")
length(intIdx)
## Rule says Dunn & Silhoutte  Should be maximum


####### End of Checking all Criteria with K=3 #########
########################################################




############### For K=4 ########################################
################################################################

Rprof()
res.kmeans.4 <- kmeans(df,
                       centers = 4,
                       nstart = 100,
                       iter.max = 300)
print(res.kmeans.4)
Rprof(NULL)

summaryRprof()

############## OUTPUT ##################

# K-means clustering with 4 clusters of sizes 3897, 31748, 11916, 59945

"Cluster means:
   Recency Frequency  Monetary
1 1.278676 21.950218 174.47111
2 1.872401  8.262316  44.43273
3 1.559080 15.201410  92.20554
4 1.989190  3.701126  15.33188"

###################### Check for percentage accurate partition
################### between_ss / total_SS
"Within cluster sum of squares by cluster:
[1] 4376380 4338973 4708938 3618758
 (between_SS / total_SS =  89.3 %)"

######## Timing ##########

"$by.self
                     self.time self.pct total.time total.pct
.Fortran               10.22    93.76      10.22     93.76
array                   0.22     2.02       0.22      2.02
asplit                  0.14     1.28       0.36      3.30
print.default           0.10     0.92       0.10      0.92
integer                 0.06     0.55       0.06      0.55
do_one                  0.04     0.37      10.34     94.86
%in%                    0.02     0.18       0.02      0.18
do.call                 0.02     0.18       0.02      0.18
duplicated.default      0.02     0.18       0.02      0.18
get                     0.02     0.18       0.02      0.18
sample.int              0.02     0.18       0.02      0.18
sys.call                0.02     0.18       0.02      0.18"

# $sampling.time
# [1] 10.9


####### Plotting the clusters using PCA technique ########

# Dimension reduction using PCA
res.pca <- prcomp(df_RFM2[,-1],  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.kmeans.4$cluster)
# Add Species groups from the original data sett

# Data inspection
head(ind.coord)


# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "euclid",
  repel = TRUE,
  shape = "cluster", size = 1.5,  legend = "right", ggtheme = theme_minimal(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
# Cluster Graph Shows small amount of overlapping

##### Print the means of original data columns with clusters =2
dd <- cbind(df, cluster = res.kmeans.4$cluster)
temp        <- aggregate(dd, by=list(res.kmeans.4$cluster), FUN=mean)
temp2       <- t(temp)
temp2

"      [,1]      [,2]     [,3]      [,4]
Group.1     1.000000  2.000000  3.00000  4.000000
Recency     1.278676  1.872401  1.55908  1.989190
Frequency  21.950218  8.262316 15.20141  3.701126
Monetary  174.471114 44.432726 92.20554 15.331881
cluster     1.000000  2.000000  3.00000  4.000000"

" Cluster 1 = Loyal Customers
Cluster 2 = new customers
Cluster 3 = Loyal Customers Level-2
Cluster 4 = Customers at Risk"

##### Changing the dimensionality to check the performance of K =4
### Remove the column Recency and check the quality of Kmeans

df1 = df[2:3]
str(df1)
Rprof()
res.kmeans.4_1 <- kmeans(df1,
                         centers = 4,
                         nstart = 100,
                         iter.max = 300)
print(res.kmeans.4_1)
Rprof(NULL)

summaryRprof()

############## OUTPUT ##################

"K-means clustering with 4 clusters of sizes 59951, 11915, 31743, 3897

Cluster means:
  Frequency  Monetary
1  3.701506  15.33331
2 15.201511  92.20757
3  8.262641  44.43626
4 21.950218 174.47111"

"Within cluster sum of squares by cluster:
[1] 3502991 4686108 4276126 4369631
 (between_SS / total_SS =  89.4 %)"

"$by.self
                     self.time self.pct total.time total.pct
.Fortran                7.64    93.17       7.64     93.17
array                   0.22     2.68       0.26      3.17
asplit                  0.14     1.71       0.40      4.88
print.default           0.08     0.98       0.08      0.98
duplicated.default      0.04     0.49       0.04      0.49
is.atomic               0.04     0.49       0.04      0.49
do_one                  0.02     0.24       7.66     93.41
$                       0.02     0.24       0.02      0.24"

# $sampling.time
# [1] 8.2

# Not much difference in terms of partition

####### Plotting the clusters using PCA technique ########

# Dimension reduction using PCA

res.pca <- prcomp(df_RFM2[,c(-1,-3)],  scale = TRUE)
# Coordinates of individuals
ind.coord <- as.data.frame(get_pca_ind(res.pca)$coord)
# Add clusters obtained using the K-means algorithm
ind.coord$cluster <- factor(res.kmeans.4_1$cluster)
# Add Species groups from the original data sett

# Data inspection
head(ind.coord)


# Percentage of variance explained by dimensions
eigenvalue <- round(get_eigenvalue(res.pca), 1)
variance.percent <- eigenvalue$variance.percent
head(eigenvalue)

ggscatter(
  ind.coord, x = "Dim.1", y = "Dim.2", 
  color = "cluster", palette = "npg", ellipse = TRUE, ellipse.type = "euclid",
  repel = TRUE,
  shape = "cluster", size = 1.5,  legend = "right", ggtheme = theme_minimal(),
  xlab = paste0("Dim 1 (", variance.percent[1], "% )" ),
  ylab = paste0("Dim 2 (", variance.percent[2], "% )" )
) +
  stat_mean(aes(color = cluster), size = 4)
##### Print the means of original data columns with clusters = 4
dd <- cbind(df1, cluster = res.kmeans.4$cluster)
temp        <- aggregate(dd, by=list(res.kmeans.4$cluster), FUN=mean)
temp2       <- t(temp)
temp2

## In terms of graph(overlapping) it increases and grouping there is not much difference#


#### Checking the Internal Indices on first K = 4 #####


getCriteriaNames(TRUE)
intIdx <- intCriteria(as.matrix(df),res.kmeans.2$cluster,"Dunn")
length(intIdx)

intIdx <- intCriteria(as.matrix(df),res.kmeans.2$cluster,"Silhouette")
length(intIdx)
## Rule says Dunn & Silhoutte  Should be maximum


####### End of Checking all Criteria with K=4 #########
########################################################



"Result= As the silhoute width is max for k =4 and the overlapping is less
in K =4 also the percentage partition is 89.4 % the optimal K = 4 is selected"


##### After this please refer to jupyter notebook
### Foe Hyperparameter tuning with K =4 #######

##########################################################################






