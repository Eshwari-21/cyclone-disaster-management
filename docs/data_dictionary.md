# Data Dictionary â€” Cyclone Disaster Management

## 1. Project Scope

### Study Area

The spatial study area consists of five contiguous coastal districts of Andhra Pradesh:

- Srikakulam
- Vizianagaram
- Visakhapatnam
- Anakapalli
- Kakinada

### Common Spatial Unit

All spatial datasets are harmonized to a project-defined **1 km Ã— 1 km grid** covering the study area.

The grid contains **17,737 cells**.

The grid is created in **UTM Zone 44N (EPSG:32644)** so that the 1 km cell size is defined in metres. The final grid is stored in **WGS 84 (EPSG:4326)** for compatibility with geographic datasets and GIS workflows.

The 1 km grid is the project's common spatial integration unit. It does not imply that every source dataset has a native spatial resolution of 1 km.

### Primary Processed Dataset

The main integrated grid-level dataset is:

`data/processed/master_grid_2024.csv`

It contains one record per grid cell and combines:

- environmental variables,
- population exposure,
- spatial features derived from 2024 IMD cyclone events.

The dataset contains **17,737 rows and 26 columns**.

### Important Scope Note

The current processed data provide spatial predictor and exposure features. No independently observed historical disaster-impact target variable has been established in the current project files.

Therefore, `master_grid_2024.csv` should currently be treated as a **feature/exposure dataset**, not as a supervised machine-learning training dataset with an established target.
## 2. Source Dataset Inventory

The project uses multiple heterogeneous data sources. Each source is processed separately before being harmonized to the common 1 km grid.

| Source | Dataset / Product | Temporal Coverage Used | Main Purpose | Raw Location |
|---|---|---|---|---|
| Administrative boundaries | Andhra Pradesh district boundaries | Current boundary dataset | Define the five-district study area | `data/raw/boundaries/` |
| IMD | 2024 Best Track | 2024 | Cyclone/depression track and intensity information | `data/raw/imd/` |
| GEE | Elevation | 2024-derived raster | Terrain/elevation feature | `data/raw/gee/` |
| GEE | Slope | 2024-derived raster | Terrain/slope feature | `data/raw/gee/` |
| GEE | Rainfall | 2024-derived raster | Environmental rainfall feature | `data/raw/gee/` |
| GEE | NDVI | 2024-derived raster | Vegetation/environmental feature | `data/raw/gee/` |
| WorldPop | Population 1 km Aggregated UN-adjusted | 2020 | Population exposure | `data/raw/population/` |
| MOSDAC | 3RIMG L2G GPI | 30 Augâ€“1 Sep 2024 | Time-varying GPI observations | `data/raw/mosdac/` |

### Source Preservation

Raw source files are preserved under `data/raw/` and are not overwritten by preprocessing operations.

Intermediate transformations are stored under `data/interim/`.

Cleaned, spatially harmonized, and integrated outputs are stored under `data/processed/`.

This separation allows each processing stage to be reproduced and independently validated.

### Temporal Considerations

The source datasets do not all represent the same time period:

- WorldPop population data used in the project are from 2020.
- GEE environmental layers are derived for 2024.
- IMD cyclone-event features are derived from the 2024 Best Track dataset.
- MOSDAC GPI observations used in the project cover 30 Augustâ€“1 September 2024.

These temporal differences are retained explicitly rather than treating all variables as observations from a single date.

## 3. Common Spatial Grid

### Purpose

A common spatial grid is used to integrate datasets that have different spatial formats, coordinate systems, and native resolutions.

The project-defined spatial unit is a **1 km Ã— 1 km grid** covering the five-district study area.

Each grid cell is identified by a unique `grid_id`.

### Grid Creation

The study-area district boundaries were first transformed to **UTM Zone 44N (EPSG:32644)**.

A regular 1 km Ã— 1 km fishnet was created in this projected coordinate system because distances and cell dimensions are represented in metres.

The generated cells were then clipped to the five-district study-area boundary.

After spatial processing, the grid was converted to **WGS 84 (EPSG:4326)** for compatibility with the other spatial datasets.

### Grid Output

The final grid is stored as:

`data/processed/coastal_ap_grid_1km.gpkg`

Grid validation produced:

- Total grid cells: **17,737**
- CRS: **EPSG:4326**
- Unique `grid_id` values: **17,737**
- Duplicate grid IDs: **0**
- Empty geometries: **0**
- Missing geometries: **0**
- Invalid geometries: **0**

### Boundary Cells

Cells along district boundaries may have an area smaller than 1 kmÂ² because the regular 1 km Ã— 1 km cells are clipped by the irregular study-area boundary.

Therefore, the phrase "1 km grid" refers to the intended regular cell size before boundary clipping. Boundary cells are not expected to have exactly 1 kmÂ² of retained area.

### Role in Data Integration

The grid acts as the common spatial reference for the processed datasets.

Examples include:

- Environmental raster values â†’ grid cells
- Population raster values â†’ grid cells
- IMD cyclone-track information â†’ spatial distance features for grid cells
- MOSDAC GPI observations â†’ grid-associated time-series observations

The common grid allows heterogeneous datasets to be joined using `grid_id`.

### Important Resolution Note

The common grid does **not** change the native spatial resolution of a source dataset.

For example, a coarse rainfall or GPI observation can be associated with multiple 1 km grid cells. The resulting records are therefore grid-level representations of the source data, not evidence that the original source had 1 km spatial resolution.

## 4. Environmental Data Preprocessing

Environmental variables were obtained as raster datasets and harmonized to the common 1 km project grid.

The environmental variables used in the integrated dataset are:

- Elevation
- Slope
- Rainfall
- NDVI

The processed environmental dataset is:

`data/processed/gee_environmental_grid_2024.csv`

It contains one row per project grid cell.

### 4.1 Elevation

Source file:

`data/raw/gee/coastal_ap_elevation_2024.tif`

Output column:

`elevation_m`

Unit:

metres (m)

The raster was spatially summarized for each 1 km grid cell to obtain a grid-level elevation value.

Validation of the processed values produced:

- Minimum: 0 m
- Maximum: approximately 1081.69 m
- Missing grid cells: 23

### 4.2 Slope

Source file:

`data/raw/gee/coastal_ap_slope_2024_final.tif`

Output column:

`slope_deg`

Unit:

degrees

The slope raster was spatially summarized for each project grid cell.

Validation of the processed values produced:

- Minimum: 0Â°
- Maximum: approximately 40.94Â°
- Missing grid cells: 23

### 4.3 Rainfall

Source file:

`data/raw/gee/coastal_ap_rainfall_2024.tif`

Output column:

`annual_rainfall_mm`

Unit:

millimetres (mm)

The rainfall raster has a substantially coarser native spatial resolution than the project 1 km grid.

Because a simple point-based raster lookup left a large number of grid cells without values, a separate spatial mapping procedure was used.

Valid rainfall raster pixels were represented spatially and intersected with the project grid. For each grid cell, the rainfall value from the source pixel with the largest overlap area was assigned.

The resulting dataset is:

`data/processed/rainfall_grid_2024.csv`

Validation produced:

- Valid grid cells: 17,687
- Missing grid cells: 50
- Minimum: approximately 948.58 mm
- Maximum: approximately 1865.62 mm

The 50 missing values were retained as missing because the source raster did not provide valid rainfall values at those locations. They were not replaced with zero.

### 4.4 NDVI

Source file:

`data/raw/gee/coastal_ap_ndvi_2024.tif`

Output column:

`mean_ndvi`

NDVI is a dimensionless vegetation index.

The NDVI raster was spatially summarized for each project grid cell using an all-touched zonal-statistics approach.

The resulting dataset is:

`data/processed/ndvi_grid_2024.csv`

Validation produced:

- Valid grid cells: 17,642
- Missing grid cells: 95
- Minimum: approximately -0.1347
- Maximum: approximately 0.8289

The 95 missing values were retained as missing because the source raster did not provide valid NDVI observations at those locations.

### 4.5 Environmental Integration

The environmental variables were combined using `grid_id` as the spatial key.

The resulting dataset is:

`data/processed/gee_environmental_grid_2024.csv`

It contains:

- `grid_id`
- `elevation_m`
- `slope_deg`
- `annual_rainfall_mm`
- `mean_ndvi`

The environmental dataset contains **17,737 rows**, corresponding to the complete project grid.

### Missing-Value Policy

Missing environmental values are retained as `NaN` during preprocessing.

They are not automatically replaced with zero or a statistical estimate at this stage because the appropriate missing-value treatment depends on the subsequent analysis or modeling method.    
## 5. Population Data Preprocessing

Population is used as an exposure variable because the potential impact of a cyclone can depend on the number of people located within an affected area.

### Source

Raw population raster:

`data/raw/population/ind_ppp_2020_1km_Aggregated_UNadj.tif`

The population dataset represents a 2020 population estimate and should not be interpreted as an official census count.

### Spatial Processing

The population raster was spatially summarized over the project 1 km grid using zonal statistics.

For each grid cell, the population values intersecting the cell were aggregated to produce a grid-level population estimate.

The processing retained the population sum as the primary exposure variable.

### Output

Processed population dataset:

`data/processed/population_grid_2020.csv`

Primary output column:

`population_sum`

The dataset contains:

- 17,737 grid cells
- 17,737 unique `grid_id` values
- 0 missing population values

### Population Summary

The processed population values have:

- Minimum: 0
- Median: approximately 1,304.24
- Mean: approximately 2,160.44
- Maximum: approximately 110,173.55

The estimated total population represented across the study-area grid is approximately **38.32 million**.

### Integration

Population was integrated with the environmental grid dataset using:

`grid_id`

The resulting population value is stored in:

`data/processed/master_grid_base_2024.csv`

and subsequently in:

`data/processed/master_grid_2024.csv`

### Temporal Consideration

The population data are from 2020, while the environmental and IMD-derived features used in the current master dataset are associated with 2024.

Therefore, `population_sum` should be interpreted as a **2020 population exposure estimate used alongside the 2024 spatial features**, rather than as a 2024 population count.
## 6. IMD Cyclone Track Preprocessing

The India Meteorological Department (IMD) Best Track dataset was used to obtain historical cyclone and depression track information for 2024.

### Source

Raw IMD Best Track document:

`data/raw/imd/33_82f41b_33_4d5ff5_Best_Track_2024_ (1).pdf`

The document contains Best Track information for 13 named or classified 2024 systems/events represented in the processed dataset.

### Track Extraction

The PDF tables were parsed into a structured tabular dataset containing track observations.

The extraction process handles:

- event identification,
- observation dates,
- observation times,
- latitude,
- longitude,
- central pressure,
- pressure change,
- maximum sustained wind speed,
- cyclone intensity/category,
- cyclone identification information where available.

The cleaned observation-level output is:

`data/processed/imd_best_track_2024_clean.csv`

### Track Validation

The cleaned dataset contains:

- 251 track observations
- 13 events
- No duplicate event/timestamp combinations
- Latitude range: 5.0Â° to 28.8Â°
- Longitude range: 58.8Â° to 91.8Â°
- No invalid latitude/longitude values
- No negative time gaps within an event
- No within-event gaps greater than 24 hours

The `ci_no` field contains missing values for some observations. Other primary track fields are complete in the validated dataset.

### Spatial Association with the Project Grid

The complete IMD track dataset was spatially joined against the project grid.

Only track observations that physically fall inside the five-district study area receive a direct grid-cell association.

Most IMD track observations lie outside the study area. Therefore, they are not artificially forced into the study-area grid.

The direct point-to-grid output is:

`data/processed/imd_best_track_2024_grid.csv`

This contains the direct spatial matches between IMD track points and the project grid.

### Distance-Based Grid Features

Because cyclone tracks frequently pass outside the study area while still influencing nearby locations, distance-based features were generated for every project grid cell.

For each combination of:

- project grid cell, and
- 2024 IMD event,

the minimum distance from the grid-cell centroid to the event's track observations was calculated.

The resulting dataset is:

`data/processed/imd_grid_event_features_2024.csv`

It contains one record for each grid-cell/event combination.

With:

- 17,737 grid cells
- 13 IMD events

the dataset contains:

**230,581 grid-cell/event records.**

### Grid-Level IMD Summary

The event-level distance information was summarized into:

`data/processed/imd_grid_features_2024.csv`

and subsequently integrated into:

`data/processed/master_grid_2024.csv`

The grid-level IMD-derived variables include:

- `distance_event_1_km` through `distance_event_13_km`
- `min_event_distance_km`
- `distance_to_strongest_event_km`
- `events_within_50km`
- `events_within_100km`

### Interpretation

The distance variables represent the spatial relationship between each grid cell and the 2024 IMD event tracks.

They should therefore be interpreted as **2024 historical-event spatial features**, not as a general long-term cyclone climatology.

### Important Feature Note

The following event-summary fields currently have no spatial variation:

- `strongest_event_id`
- `max_event_msw_kt`

The strongest event in the current 13-event dataset is the same for every grid cell under the current aggregation logic.

These fields are retained in the comprehensive master dataset for traceability, but they are not spatially informative as grid-level predictors in their current form.
## 7. MOSDAC GPI Preprocessing

MOSDAC INSAT-3DR GPI observations are used as a time-varying environmental dataset.

### Source

The raw GPI files are stored under:

`data/raw/mosdac/`

The processed files used for this pipeline follow the naming pattern:

`3RIMG_*_L2G_GPI_*.h5`

The GPI observations used in the current pipeline cover:

**30 August 2024 to 1 September 2024**

with 15 observation timestamps.

### Raw HDF5 Structure

The GPI HDF5 files contain the following datasets:

- `GPI`
- `latitude`
- `longitude`
- `time`

The GPI array has the structure:

`(1, latitude, longitude)`

The single time dimension is removed during preprocessing so that the GPI values can be associated with the latitude and longitude arrays.

The GPI values are stored as floating-point values with units reported as millimetres (`mm`).

### Missing-Value Handling

The source GPI dataset uses `-999` as the fill value.

During preprocessing:

- `-999` is converted to `NaN`.
- `0` is retained as a valid GPI value.

This distinction is important because zero represents a valid zero-valued observation and must not be interpreted as missing data.

### Observation-Level Output

The raw HDF5 observations are converted into a tabular intermediate dataset:

`data/interim/mosdac/mosdac_gpi_points.csv`

The observation-level structure is:

- `timestamp_utc`
- `latitude`
- `longitude`
- `gpi_mm`

Only observations within the study-area bounding box are retained at this stage.

### Spatial Mapping to the Project Grid

The GPI observations are mapped to the project 1 km grid using:

`src/spatial/map_gpi_to_grid.py`

The GPI latitude/longitude coordinates are treated as the centers of 1Â° Ã— 1Â° source cells for the spatial association step.

The centroids of the project 1 km grid cells are calculated in the projected CRS EPSG:32644 and then transformed to EPSG:4326.

Each grid-cell centroid is associated with the GPI source cell containing that centroid.

This avoids assigning a GPI observation to every 1 km cell that merely touches the boundary of a coarse source cell.

### Grid-Level Time Series Output

The resulting dataset is:

`data/processed/mosdac_gpi_grid_2024.csv`

It contains:

- `grid_id`
- `timestamp_utc`
- `gpi_latitude`
- `gpi_longitude`
- `gpi_mm`

The validated output contains:

- 242,295 mapped observations
- 16,153 unique grid cells
- 15 unique timestamps
- 0 duplicate grid/timestamp combinations
- GPI range: 0 to 9 mm

Every grid cell that receives a GPI assignment has observations at all 15 timestamps.

Therefore:

- Grid cells with GPI coverage: 16,153
- Total project grid cells: 17,737
- Grid cells without a GPI assignment under the current spatial mapping rule: 1,584

### Temporal Aggregation

The current pipeline preserves the GPI observations as a time series.

No mean, maximum, cumulative, or other temporal aggregation is currently incorporated into `master_grid_2024.csv`.

This is intentional because the appropriate temporal aggregation depends on the intended analytical use and should not be selected arbitrarily during preprocessing.

### Important Resolution Note

The GPI source is substantially coarser than the project 1 km grid.

Mapping GPI observations onto the 1 km grid does not increase the native spatial resolution of the GPI product.

Multiple 1 km grid cells can therefore receive the same GPI observation.

### Important Assumption

The current spatial mapping treats each GPI latitude/longitude coordinate as the center of a 1Â° Ã— 1Â° source cell.

This is an explicit processing assumption used by the project mapping script and should not be interpreted as an independently verified statement about the source product's coordinate semantics.

## 8. Master Grid Integration

The processed environmental, population, and IMD-derived datasets are integrated using the common `grid_id`.

The objective of this stage is to create one comprehensive record for each project grid cell.

### Integration Sequence

The integration is performed in two stages.

#### Stage 1 â€” Environmental + Population Integration

The following datasets are combined:

`data/processed/gee_environmental_grid_2024.csv`

and

`data/processed/population_grid_2020.csv`

using:

`grid_id`

The result is:

`data/processed/master_grid_base_2024.csv`

This dataset contains:

- `grid_id`
- `elevation_m`
- `slope_deg`
- `annual_rainfall_mm`
- `mean_ndvi`
- `population_sum`

#### Stage 2 â€” IMD Feature Integration

The IMD grid-level feature dataset:

`data/processed/imd_grid_features_2024.csv`

is then joined to `master_grid_base_2024.csv` using:

`grid_id`

The final integrated dataset is:

`data/processed/master_grid_2024.csv`

### Final Dataset Structure

The final master dataset contains:

- 17,737 rows
- 26 columns
- 17,737 unique grid IDs
- 0 duplicate grid IDs

Each row represents one spatial grid cell.

### Feature Groups

The final dataset contains the following feature groups.

#### Spatial Identifier

- `grid_id`

#### Environmental Features

- `elevation_m`
- `slope_deg`
- `annual_rainfall_mm`
- `mean_ndvi`

#### Population Exposure

- `population_sum`

#### IMD Event-Distance Features

- `distance_event_1_km`
- `distance_event_2_km`
- `distance_event_3_km`
- `distance_event_4_km`
- `distance_event_5_km`
- `distance_event_6_km`
- `distance_event_7_km`
- `distance_event_8_km`
- `distance_event_9_km`
- `distance_event_10_km`
- `distance_event_11_km`
- `distance_event_12_km`
- `distance_event_13_km`

#### IMD Summary Features

- `min_event_distance_km`
- `strongest_event_id`
- `strongest_event_name`
- `max_event_msw_kt`
- `distance_to_strongest_event_km`
- `events_within_50km`
- `events_within_100km`

### Master Dataset Validation

The final master dataset was validated for:

- expected row count,
- unique grid IDs,
- duplicate records,
- missing values,
- numeric ranges,
- zero-variance columns.

Validation confirmed:

- 17,737 unique grid cells
- 0 duplicate grid IDs
- Complete IMD distance features
- Complete population values
- Environmental missing values retained from source-data coverage

### Missing Environmental Values

The final dataset contains missing values in:

- `elevation_m`: 23
- `slope_deg`: 23
- `annual_rainfall_mm`: 50
- `mean_ndvi`: 95

These missing values are preserved rather than automatically imputed.

The appropriate treatment is deferred until the downstream analytical or modeling requirements are established.

### MOSDAC Separation

The MOSDAC GPI time series is not currently merged into the one-row-per-grid master dataset.

This is because MOSDAC contains multiple observations per grid cell across time, whereas the master dataset contains one record per grid cell.

The MOSDAC time series is therefore maintained separately as:

`data/processed/mosdac_gpi_grid_2024.csv`

This preserves the temporal information and avoids applying an undocumented temporal aggregation.

### Data-Engineering Endpoint

`master_grid_2024.csv` is the current integrated spatial feature dataset produced by the data-engineering pipeline.

It should not be interpreted as a labeled machine-learning dataset because an independently observed disaster-impact target has not yet been established in the current project data.

## 9. Validation, Assumptions, Limitations, and Reproducibility

### 9.1 Validation

Validation was performed at multiple stages of the preprocessing pipeline.

The validation checks included:

- file and dataset availability,
- coordinate reference systems,
- geometry validity,
- duplicate identifiers,
- missing values,
- numeric ranges,
- spatial coverage,
- temporal coverage,
- duplicate timestamps,
- grid-to-source associations.

The final master grid was validated to contain:

- 17,737 grid cells
- 17,737 unique `grid_id` values
- 0 duplicate grid IDs

### 9.2 Environmental Data Validation

Environmental variables were checked for missing values and plausible numeric ranges.

The final missing-value counts are:

| Variable | Missing |
|---|---:|
| `elevation_m` | 23 |
| `slope_deg` | 23 |
| `annual_rainfall_mm` | 50 |
| `mean_ndvi` | 95 |

These missing values originate from source-data coverage and are preserved during preprocessing.

### 9.3 Population Validation

The population grid was validated for:

- complete grid coverage,
- unique grid IDs,
- missing population values,
- numeric ranges.

The final population dataset contains 17,737 unique grid cells and no missing population values.

### 9.4 IMD Validation

The cleaned IMD track dataset was checked for:

- duplicate event/timestamp combinations,
- valid latitude and longitude ranges,
- temporal ordering,
- missing primary fields,
- event-level row counts.

The final cleaned dataset contains 251 observations across 13 events.

The derived grid-event feature dataset contains exactly one record for each grid/event combination:

17,737 grid cells Ã— 13 events = 230,581 records.

### 9.5 MOSDAC Validation

The MOSDAC grid-level dataset was checked for:

- number of timestamps,
- GPI value range,
- duplicate grid/timestamp combinations,
- temporal coverage per grid cell.

The mapped dataset contains:

- 16,153 grid cells with GPI coverage
- 15 timestamps
- 242,295 grid/timestamp observations
- 0 duplicate grid/timestamp combinations

Every grid cell receiving GPI coverage has all 15 timestamps.

### 9.6 Important Processing Assumptions

The current pipeline includes several explicit assumptions.

#### Common Grid

A project-defined 1 km grid is used as the common spatial integration unit.

#### GPI Source Cells

GPI latitude/longitude coordinates are treated as the centers of 1Â° Ã— 1Â° source cells for the spatial mapping procedure.

#### Grid Centroids

For GPI spatial association, centroids of project grid cells are calculated in EPSG:32644 before being transformed to geographic coordinates.

#### IMD Distance Features

Cyclone-track influence is represented spatially using the distance between grid-cell centroids and 2024 IMD event tracks.

#### Temporal Representation

MOSDAC GPI observations are retained as a time series. No undocumented temporal aggregation is applied.

### 9.7 Known Limitations

#### Limited IMD Time Period

The current IMD spatial features are derived from 2024 Best Track events only.

They should therefore not be interpreted as a long-term cyclone climatology.

#### Population Temporal Mismatch

The population dataset represents 2020, while several environmental and cyclone-derived variables represent 2024.

#### Coarse Source Resolution

Mapping coarse environmental products to the 1 km grid does not increase their native spatial resolution.

Multiple project grid cells may therefore share the same source value.

#### Environmental Missing Values

Some environmental source data do not provide valid values for every project grid cell.

These missing values are currently preserved rather than imputed.

#### MOSDAC Coverage

The current MOSDAC GPI mapping provides coverage for 16,153 of the 17,737 project grid cells under the implemented spatial-association rule.

The remaining cells are not assigned a GPI value.

#### No Established Supervised Target

The current project files do not contain an independently observed historical disaster-impact target such as verified flood impact, damage, or affected population.

Therefore, the current master dataset is a feature/exposure dataset and should not yet be treated as a supervised machine-learning training dataset.

### 9.8 Reproducibility

The preprocessing pipeline is implemented as scripts under:

`src/preprocessing/`

`src/spatial/`

`src/validation/`

The main processed outputs are stored under:

`data/processed/`

Raw source data are preserved under:

`data/raw/`

Intermediate transformations are stored under:

`data/interim/`

This directory structure allows the transformation from source datasets to the integrated master grid to be reproduced and audited.

### 9.9 Current Data-Engineering Deliverable

The current data-engineering pipeline produces:

`data/processed/master_grid_2024.csv`

as the primary integrated grid-level feature dataset.

The pipeline also preserves source-specific intermediate and processed datasets so that individual transformations can be traced back to their original data source.
### Master Grid Feature Note
The fields max_event_msw_kt, strongest_event_id, and strongest_event_name are constant across the current 2024 master grid because they represent the globally strongest event among the 13 IMD events used in the dataset (DANA, event 11, with maximum MSW of 60 kt). These fields are retained for event context and provenance but should not be treated as spatially varying predictor features.


### NDVI Validation Note
Negative NDVI values were retained because negative NDVI values occur in the source raster. The raw 2024 NDVI raster ranges from -0.1918 to 0.88425, while the integrated grid values range from -0.1347 to 0.828854. Therefore, the negative grid values fall within the observed source-data range and were not treated as invalid.

### Population Validation Note
The integrated population dataset contains 98 of 17,737 grid cells (0.55%) with a population_sum of zero. These cells were retained because zero population is a valid source-derived value and was not treated as missing data.

### Final Feature Specification
The current master grid contains the following feature roles:
- Grid identifier: grid_id is the spatial key and is not a predictor.
- Environmental features: elevation_m, slope_deg, annual_rainfall_mm, and mean_ndvi.
- Exposure feature: population_sum.
- Cyclone spatial-history features: distance_event_1_km through distance_event_13_km, min_event_distance_km, and distance_to_strongest_event_km.
- Threshold proximity features: events_within_50km and events_within_100km.
- Event context fields: strongest_event_id, strongest_event_name, and max_event_msw_kt; these are currently constant across the grid and should not be treated as spatially varying predictors.

These fields constitute the current integrated spatial feature inventory, but they do not by themselves establish a supervised-learning target. No independently observed historical disaster-impact target has been established in the current project data. The IMD distance features represent spatial proximity to the 13 2024 historical events used in this dataset and should not be interpreted as a long-term cyclone climatology.
