# 01 Data Extraction and Processing
# 
# This is the first step of the workflow for my Master's thesis, titled:
# 
# "Instance-Aware Semantic Segmentation of Small Woody Landscape Features Using 
# High-Resolution Aerial Images in Biodiversity Exploratories"
# 
# This script is followed by script 02_dataset_generation.py. 
# 
# The primary tasks of this script is to extract the images from the RSDB 
# database (https://doi.org/10.1111/ecog.05266). These images will later be used
# for two purposes: 1) A version only covering the area for which mask polygons 
# exist, will be cut into smaller patches for training the models; 2) A version 
# reaching over the known masked area will be used for the final predictions of 
# the models. The extraction from the RSDB database is done through extracting a 
# raster rgb nir 32 bit image for each plot in the three biodiversity explorato-
# ries. Each image has the size of the largest square fitting into the circular 
# plot. The side length of these image squares is 706.8 m for the training data-
# set and 2000 m for the prediction maps. These are the steps performed here:
# 
# 1. Data Retrieval:
#    - biodiversity exploratory groups (POI: HAI, ALB, SCH) and raster data for
#      specific regions are extracted from the RSDB database.
# 
# 2. Data Cleaning and Transformation:
#    - Invalid POIs are removed, and SCH POIs are reprojected from EPSG:32633 to
#      EPSG:32632.
#    - Transformed POIs are reformatted and merged.
# 
# 3. Raster Data Extraction:
#    - Rasters of the two extents (706.8 m for training dataset and 2000 
#      meters for large prediction maps) are extracted for each POI and saved 
#      as .tif files.
#
# Parallel to this, the SWF polygons from atkis and updated atkis were merged 
# and filtered in QGis to serve as data for later mask creation.
# 
# In script 02_dataset_generation.py, a dataset of 256x256 pixels image patches 
# and corresponding masks is generated from the train dataset images  and
# from ATKIS-derived mask polygons representing three classes of small woody 
# landscape features.


# setup ----
# install packages
# RSDB
if(!require("remotes")) install.packages("remotes")
remotes::install_github("environmentalinformatics-marburg/rsdb/r-package")

# load packages
library(RSDB)
library(raster)
library(sf)

################################################################################
# 01 Data Retrieval ----
# set account
userpwd <- "pauline.drews:rFkzNJDfE3gQ" # use this account

# open remote sensing database
remotesensing <- RemoteSensing$new(
  "https://vhrz1078.hrz.uni-marburg.de:8201", 
  userpwd) # remote server

# get rasterdb be_dop20
rasterdb <- remotesensing$rasterdb("be_dop20")

# Check available rasters in the database for the SCH region
available_rasters <- list(rasterdb)
print(available_rasters)

# get POI groups
pois_be_hai <- remotesensing$poi_group("be_hai_poi")
pois_be_alb <- remotesensing$poi_group("be_alb_poi")
pois_be_sch <- remotesensing$poi_group("be_sch_poi")

################################################################################
# 02 Data Cleaning and Transformation ----
str(pois_be_hai)
pois_be_alb <- pois_be_alb[- c(11, 
                               31), ] # remove cancelled pois

# Transform pois_be_sch to EPSG:32632
# Convert pois_be_sch to sf object with original CRS (assuming EPSG:32633)
pois_be_sch_sf <- st_as_sf(pois_be_sch, coords = c("x", "y"), crs = 32633)

# Transform to EPSG:32632
pois_be_sch_sf <- st_transform(pois_be_sch_sf, crs = 32632)

# Check the transformed object
print(pois_be_sch_sf)

# Convert back to data.frame with correct format
pois_be_sch_df <- as.data.frame(st_coordinates(pois_be_sch_sf))
pois_be_sch_df$name <- pois_be_sch_sf$name

# Reorder columns to match the orig. format & rename x, y columns to lowercase
pois_be_sch <- pois_be_sch_df[, c("name", "X", "Y")]
names(pois_be_sch) <- c("name", "x", "y")

# Set row names to the name column
rownames(pois_be_sch) <- pois_be_sch$name

poi_list <- rbind(pois_be_hai, pois_be_alb, pois_be_sch)

################################################################################
# 03 Raster Data Extraction ----
# Function to save raster as .tif
save_raster_as_tif <- function(raster_data, output_path) {
  #if (!inherits(raster_data, "RasterLayer")) {
  #  stop("Provided data is not a RasterLayer.")
  #}
  writeRaster(
    raster_data, 
    filename = output_path, 
    format = "GTiff", 
    overwrite = TRUE)
}

# Process each POI
for (i in 238:238) {# replace 1:1 with seq_len(nrow(poi_list)) for whole data
  poi_name <- poi_list$name[i]
  poi_x <- poi_list$x[i]
  poi_y <- poi_list$y[i]
  
  # Create extent around POI
  diameter <- 2000 # 706.8 for training data, 2000 for prediction
  ext_poi <- extent_diameter(poi_x, poi_y, diameter)
  
  # Get raster of this extent
  r <- rasterdb$raster(ext_poi, time_slice = "old")
  
  # find out if 8bit 16bit or 32bit images
  raster_type <- dataType(r)
  print(paste("Raster datatype for POI", poi_name, ":", raster_type))
  
  # Check if the extent is within the raster bounds 
  print(paste("Extent bounds:", ext_poi))

  # Output image 
  output_image_path <- paste0(
    "C:/Users/Power/OneDrive/Desktop/aer_img_large_pred/", # where space enough
    "squ_", 
    poi_name, 
    ".tif")
  save_raster_as_tif(r, output_image_path)
  print(paste("Saved square of POI", poi_name, "as .tif file"))
}


# Visualize to see if Schorfheide reprojection worked
# library(raster)

# # Function to visualize a .tif image
# visualize_tif_image <- function(tif_path) {
#   # Load the .tif file as a raster object
#   tif_raster <- raster(tif_path)
#   
#   # Plot the raster
#   plot(tif_raster, main = paste("Visualization of", basename(tif_path)))
# }
# 
# 
# # Specify the path to one of the created TIFF files
# tif_image_path <- "D:/ma_dat_v02/squ_SEG01.tif"
# 
# # Visualize the selected TIFF image
# visualize_tif_image(tif_image_path)
