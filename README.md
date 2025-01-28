# Coastal-change-detection

This repository contains a Leaflet mapping application of historical coastal landcover change for New Zealand generated from optical satellite imagery in Landsat and Sentinel-2 constellations. Users can view landcover percentage change for sediment, water, and vegetation and estimated shoreline change for Instantaenous Waterline and Edge of Vegetation shoreline proxies within each hexagonal cell.

The initial national scale coastal classification is described in [Collings et al., (2022)](https://www.mdpi.com/2072-4292/14/19/4827), and a map-to-image change detection approach was used to estimate change between this classification and subsequent imagery. Google Earth Engine was used to preprocess and download imagery, via a service account. To correct estimated waterline position for tide, beach slope was acquired from [https://zenodo.org/records/7758183](https://zenodo.org/records/7758183) which was calculated by [@kvos](https://github.com/kvos) using [CoastSat.slope](https://github.com/kvos/CoastSat.slope). The [NIWA tide api](https://developer.niwa.co.nz/docs/tide-api/latest/overview) was used to acquire tide level.

Processing and results were calculated on a NeCTAR VM and a cron job is used to update results every month. 

This performs the following steps:

* Checks for and downloads cloud-free Sentinel-2 and Landsat imagery.
* Performs map-to-image change detection to estimate landcover change and shoreline position for each hexagonal cell.
* Performs tide correction. 
* Commits updated results to repository. 






## Installation 
Packages and dependencies handled by Miniforge. Check [here](https://github.com/conda-forge/miniforge) for help installing Miniforge.
To install dependencies for this repository create a new environment with the following command:

`conda create --name coastal-monitoring --file requirements.txt`

