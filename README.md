# ellishape-cli

[![PyPI version](https://badge.fury.io/py/ellishape_cli.svg)](https://badge.fury.io/py/ellishape_cli)

A command-line tool for shape analysis using Elliptic Fourier Descriptors (EFD).

## Description

Process grayscale images (with white as foreground) to calculate shape descriptors or contour
coordinates. Supports single image processing or batch processing via file lists. Outputs results in
CSV format and optionally generates visualization images.

## Installation

```bash
pip install ellishape_cli
```

* Recommended use the program in Linux.

## Usage

### Basic Command

```bash
python3 -m ellishape_cli -i input.png -out results.csv
```

### Process Multiple Files

```bash
python3 -m  ellishape_cli -I file_list.txt -method chain_code -out batch_results.csv
```

### Command Line Options

| Option          | Description                                                                | Default | Require |
|-----------------|----------------------------------------------------------------------------|---------|---------|
| -i, -input      | Input grayscale image (white=foreground)                                   | -       | Yes*    |
| -I, -input_list | Text file containing list of input paths (one per line)                    | -       | Yes*    |
| -n, -n_order    | Number of Elliptic Fourier Descriptor orders to calculate                  | 64      | No      |
| -N, -n_dots     | Number of contour points to output                                         | 512     | No      |
| -method         | Calculate method: chain code or directly pixel dots (faster but may rough) | dots    | No      |
| -skip_normalize | Disable automatic normalization of EFD coefficients                        | False   | No      |
| -out            | Output CSV file path                                                       | -       | Yes     |
| -out_image      | Generate output visualization image                                        | False   | No      |

* Either -i or -I must be provided

## Output

### CSV File Format

Contour coordinates (x,y pairs)and normalized elliptic fourier descriptor coefficients

### Visualization Image

When using -out_image, outputs an image showing original contour and reconstructed
shape from EFD coefficients.

## Examples

### Basic analysis with visualization

```bash
python3 -m ellishape_cli -i cell.png -n 32 -out cell_analysis.csv -out_image
```

### Batch processing with chain codes

```bash
python3 -m ellishape_cli -I shapes.txt -method chain_code -N 256 -out output.csv
```

### High-precision EFD calculation

```bash
python3 -m ellishape_cli -i specimen.tiff -n 128 -skip_normalize -out highres_efd.csv
```

# Tree module

Calculate distance between samples with several methods and build neighbor-joining tree.

## Usage

Calculate __Minimum Euclidean Distance__ and build trees.

```bash
python3 -m  ellishape_cli.tree -i dots.csv -min_dist -o result
```

## Command Line Options

### Core Parameters

| Option      | Description                                                        | Default | Required |
|-------------|--------------------------------------------------------------------|---------|----------|
| -i, -input  | Input CSV matrix with feature values (first column = sample names) | -       | Yes      |
| -kind       | Metadata CSV specifying sample categories (name,kind columns)      | -       | No       |
| -o, -output | Prefix for output files                                            | out     | No       |

### Distance Metrics

| Option    | Description                                                |
|-----------|------------------------------------------------------------|
| -min_dist | Minimum Euclidean distance with rotational/shift alignment |
| -h_dist   | Hausdorff distance                                         |
| -s_dist   | Shape context distance                                     |

### Advanced Options

`-pca`:    Generate PCA plot of shape features

`-no_factor`:    Disable normalization factor in distance calculations

* If no one distance metric (-min_dist/-h_dist/-s_dist) is specified, by default
  the program calculate standard Euclidean distance.

## Input Requirements

### Feature Matrix (-i)

CSV format with:

First column: Sample names

Subsequent columns: Feature values (e.g., dots coordinates)

```csv
sample,feature1,feature2,...
cell1,0.12,0.45,...
cell2,0.08,0.39,...
```

### Metadata File (-kind)

CSV format with:

First column: Sample names

Second column: Category labels

```csv
sample,category
cell1,typeA
cell2,typeB
```

## Examples

### Basic Distance Analysis

```bash
python3 -m ellishape_cli.tree -i shapes.csv -min_dist -h_dist -o cell_comparison
```

### Category-based Analysis

```bash
python3 -m ellishape_cli.tree -i features.csv -kind cell_types.csv -s_dist -pca -o typed_analysis
```