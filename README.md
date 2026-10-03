    # Cyclone Disaster Management System

A data-driven cyclone disaster-management project focused on spatial risk analysis and resource allocation for the northern coastal districts of Andhra Pradesh, India.

This repository contains the **data engineering and geospatial preprocessing pipeline** used to prepare a common spatial dataset for downstream cyclone-risk analysis and resource-allocation components.

---

## 1. Project Overview

The project integrates heterogeneous cyclone, environmental, rainfall, vegetation, topographic, and population datasets into a common **1 km × 1 km spatial grid**.

The resulting master dataset provides a consistent spatial foundation for downstream risk-analysis and resource-allocation workflows.

### Study Area

The current study area consists of five contiguous northern coastal Andhra Pradesh districts:

- Anakapalli
- Kakinada
- Srikakulam
- Visakhapatnam
- Vizianagaram

---

## 2. Data Engineering Scope

This repository focuses on:

1. Source-data organization
2. Data preprocessing and cleaning
3. Spatial transformation
4. Common-grid generation
5. Raster-to-grid integration
6. Cyclone-track-to-grid feature generation
7. Population-to-grid aggregation
8. MOSDAC GPI-to-grid mapping
9. Dataset validation
10. Master-grid integration
11. Documentation of assumptions and limitations

The downstream machine-learning/modeling stage is outside the primary scope of this repository.

---

## 3. Data Sources

The pipeline currently integrates data from the following sources:

| Source | Data | Role |
|---|---|---|
| IMD / RSMC New Delhi | 2024 cyclone best-track data | Historical cyclone-track information |
| MOSDAC / ISRO | INSAT-3DR GPI | Satellite-derived precipitation information |
| Google Earth Engine | Elevation, slope, rainfall and NDVI | Environmental variables |
| WorldPop | 2020 population raster | Population exposure |
| Study-area boundary | Five coastal Andhra districts | Spatial extent |

Source files are documented further in:

`docs/data_dictionary.md`

---

## 4. Spatial Framework

### Common Spatial Grid

A **1 km × 1 km grid** was created over the five-district study area.

The grid is generated in **UTM Zone 44N (EPSG:32644)** so that the 1 km cell dimensions are defined in metres.

The final grid is stored in **EPSG:4326** for integration with geographic datasets and spatial workflows.

### Grid Statistics

The current grid contains:

- **17,737 grid cells**
- Unique `grid_id` for every cell
- Valid geometries
- Study-area boundary clipping

Boundary cells can have areas smaller than 1 km² because they are clipped by the irregular study-area boundary.

The common grid is a **spatial integration framework**. It does not imply that every input dataset has a native spatial resolution of 1 km.

---

## 5. Processing Architecture

The data-engineering workflow follows:

```text
Source Data
    │
    ▼
Raw Data
    │
    ▼
Preprocessing
    │
    ▼
Spatial Mapping / Aggregation
    │
    ▼
Validation
    │
    ▼
Master 1-km Grid
    │
    ▼
Downstream Risk Analysis