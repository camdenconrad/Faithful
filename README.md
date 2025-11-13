# Color Adjacency Neural Generator

A C# Avalonia application that learns color adjacency patterns from images and generates new images using Wave Function Collapse (WFC) algorithm.

## Overview

This application analyzes spatial color relationships in training images by building a weighted, 8-directional adjacency graph. It then uses this learned pattern to generate entirely new images that maintain similar color adjacency characteristics to the training set.
Rule-Based Imaging (RBI) is a generative system that synthesizes new images through the probabilistic application of learned spatial, chromatic, and structural rules extracted from an input dataset. Unlike diffusion or GAN-based models, which reconstruct pixel space through gradient-based approximation, RBI operates on a semantic rule network that governs local relationships and global coherence. This results in novel images that embody the logical essence of their dataset rather than reproducing or interpolating it.

## Features

- **8-Directional Adjacency Learning**: Analyzes color relationships in all 8 directions (N, NE, E, SE, S, SW, W, NW)
- **Structural Rule Graph**: Learns structural patterns by analyzing grayscale properties (luminance, gradients, entropy)
- **Color Quantization**: Reduces color palette using K-means clustering for efficient pattern learning
- **Wave Function Collapse Generation**: Creates coherent images by iteratively collapsing superpositions based on learned constraints
- **Structure-Aware Color Placement**: Colors are placed according to both adjacency rules and structural similarity
- **Modern UI**: Clean, intuitive Avalonia-based interface with real-time feedback
- **Configurable Parameters**: Adjustable color quantization level and output image dimensions

## How It Works

### 1. Build Structural Rule Graph

Before learning color adjacencies, the system builds a structural understanding of the training images:

- **Convert to Grayscale**: Each training image is converted to grayscale using luminance: `L = 0.299R + 0.587G + 0.114B`
- **Compute Gradient Magnitude**: Sobel filters detect edges and structural boundaries
- **Calculate Local Entropy**: Measures diversity of neighboring intensities (texture complexity)
- **Quantize Structure Classes**: K-means clustering groups pixels with similar structural properties
- **Map Colors to Structures**: Records which colors appear in which structural contexts

This ensures that areas with similar grayscale characteristics maintain similar color patterns in generated images.

### 2. Training Phase
- Load images from a selected folder
- Apply color quantization to reduce unique colors (default: 128 colors)
- Extract 8-directional adjacency patterns from each pixel
- Build weighted adjacency graph: `adjacency[color][direction][neighbor] = frequency`
- Normalize frequencies to probability distributions

### 2. Generation Phase
- Initialize output grid with all colors as possibilities for each pixel
- Iteratively collapse pixels using minimum entropy heuristic:
  - Select pixel with fewest possible colors
  - Choose color based on weighted compatibility with neighbors
  - Propagate constraints to neighboring pixels
- Continue until all pixels are determined

## Usage

### Installation

1. Ensure you have .NET 9.0 SDK installed
2. Clone this repository
3. Build the project:
   ```bash
   dotnet build
   ```

### Running the Application
