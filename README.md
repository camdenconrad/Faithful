# NNImage - AI-Powered Image Upscaling & Enhancement

**If you care what the original artist actually drew — use this one.**

## Why RepliKate Beats the Competition

| Input Type | Your RepliKate Upscaler | ESRGAN / SwinIR / etc. | Winner        |
|------------|-------------------------|------------------------|---------------|
| **Pixel art** | 100% perfect, zero smearing | Melts it into watercolor | **repliKate** |
| **Hand-painted game textures** | Preserves every intended stroke | Adds fake pores, fake grit | **repliKate**       |
| **256-512px classic 3D textures** | Looks like the artist painted it at 4K | Hallucinates new details | **repliKate**       |
| **PS1 / N64 / early 2000s assets** | Restores original intent flawlessly | Turns blocky polygons into plastic skin | **repliKate**       |
| **Actual low-res photographs** | GIGO — keeps the big ugly pixels | Invents plausible but fake detail | **ESRGAN**    |

The RepliKate Advantage: This upscaler does one amazing thing — it ONLY upscales. It doesn't add non-existent detail or smooth images. It takes your image and makes it 8x bigger (or whatever size) without hallucinating details. It's exceptional at game textures and things ML normally fails at like snow and foam. It's just the same image but high fidelity — it doesn't look stretched, it just looks... the same.

## What This Actually Does

NNImage combines several different AI approaches to upscale images properly. Unlike most upscalers that either blur everything or invent fake details, this one tries to figure out what the original artist actually intended.

## Main Features

### Progressive Upscaling
Instead of jumping straight from 1x to 4x (which usually looks terrible), this does multiple smaller steps: 1x → 1.25x → 1.5x → 2x → 4x. Each step builds on the last one.

There's also a "1x mode" that just cleans up the image without making it bigger - great for fixing compression artifacts and general cleanup.

### The Three AI Systems

**NNImage Multi-Scale Context Graphs**: This is the main workhorse. It learns color relationships and spatial patterns from your image, then applies them intelligently during upscaling. It can use GPU acceleration and handles 8 different directions of pattern recognition.

**RepliKate Sequence Prediction**: This one specializes in edges and fine details. It learns from sequences in your image to predict what pixels should look like. Perfect for text, sharp edges, and intricate details that other systems mess up.

**Intelligent Routing**: The system automatically decides which method to use for each pixel:
- About 40% gets basic bilinear interpolation (for smooth areas)
- About 50% gets the NNImage treatment (moderate detail)  
- About 10% gets RepliKate (high detail edges and textures)

### Artifact Detection and Handling

The system can tell the difference between real image detail and compression artifacts. It identifies JPEG compression blocks, analyzes multi-scale consistency, and protects genuine textures while removing junk.

When hyper-detailing is enabled, it runs a 5-pass enhancement:
1. Conservative GPU enhancement on genuine detail areas only
2. RepliKate micro-details for structured edges
3. Lighter GPU refinement pass
4. Adaptive sharpening with artifact protection  
5. Surgical smoothing that only removes confirmed artifacts

## Image Generation Capabilities

### Color Adjacency Neural Generator

NNImage includes a sophisticated image generation system that learns spatial color relationships from training images and creates entirely new images using Wave Function Collapse algorithms.

**Rule-Based Imaging (RBI)**: This is a generative system that synthesizes new images through probabilistic application of learned spatial, chromatic, and structural rules extracted from input datasets. Unlike diffusion or GAN-based models that reconstruct pixel space through gradient approximation, RBI operates on a semantic rule network governing local relationships and global coherence. The result is novel images that embody the logical essence of their training data rather than just reproducing or interpolating it.

### How Image Generation Works

#### Building the Structural Rule Graph

Before learning color adjacencies, the system builds structural understanding of training images:

- **Grayscale conversion**: Each training image converts to grayscale using luminance formula `L = 0.299R + 0.587G + 0.114B`
- **Gradient computation**: Sobel filters detect edges and structural boundaries
- **Local entropy calculation**: Measures diversity of neighboring intensities (texture complexity)
- **Structure class quantization**: K-means clustering groups pixels with similar structural properties  
- **Color-to-structure mapping**: Records which colors appear in which structural contexts

This ensures areas with similar grayscale characteristics maintain similar color patterns in generated images.

#### 8-Directional Adjacency Learning

The system analyzes color relationships in all 8 directions: North, Northeast, East, Southeast, South, Southwest, West, Northwest. It builds weighted adjacency graphs recording how often each color appears next to every other color in each direction.

#### Training Process

1. Load images from selected folder
2. Apply color quantization to reduce unique colors (default: 128 colors)
3. Extract 8-directional adjacency patterns from each pixel
4. Build weighted adjacency graph: `adjacency[color][direction][neighbor] = frequency`
5. Normalize frequencies to probability distributions

#### Generation Process

1. Initialize output grid with all colors as possibilities for each pixel
2. Iteratively collapse pixels using minimum entropy heuristic:
   - Select pixel with fewest possible colors
   - Choose color based on weighted compatibility with neighbors
   - Propagate constraints to neighboring pixels
3. Continue until all pixels are determined

### Generation Features

- **8-Directional Adjacency Learning**: Analyzes color relationships in all directions
- **Structural Rule Graph**: Learns structural patterns through grayscale analysis
- **Color Quantization**: Uses K-means clustering for efficient pattern learning
- **Wave Function Collapse Generation**: Creates coherent images by iteratively collapsing superpositions
- **Structure-Aware Color Placement**: Colors placed according to both adjacency rules and structural similarity
- **Configurable Parameters**: Adjustable color quantization level and output dimensions

### AI-Powered Image Synthesis

Beyond the adjacency generator, the upscaling system can also generate new image content:

- **Context-aware generation**: Uses trained neural networks to generate new image content
- **Pattern completion**: Fills missing or damaged areas with contextually appropriate details
- **Style consistency**: Maintains visual coherence across generated regions
- **Multi-scale synthesis**: Generates details at multiple resolution levels

### Content-Aware Enhancement

The enhancement side handles:
- **Detail hallucination**: Intelligently adds plausible details during upscaling
- **Edge reconstruction**: Recreates sharp edges and fine lines
- **Texture synthesis**: Generates consistent textures and patterns
- **Color harmony**: Maintains color relationships and gradients

## Technical Details

### Performance & Hardware Utilization

**CPU Performance:**
- Uses up to 95% of available CPU cores for maximum throughput
- Optimized parallel processing with efficient work distribution
- Typical performance: 2-5 million pixels/second depending on hardware
- Progressive upscaling is 3-5x faster than single-pass due to effective caching

**GPU Acceleration (CUDA):**
- Full CUDA support with ILGPU framework for compatible NVIDIA GPUs
- Automatic GPU memory management and optimal thread group sizing
- Pre-compiled kernels for zero warmup time
- Falls back to CPU automatically if GPU unavailable
- Bulk pattern training: up to 2 million patterns/second on modern GPUs

**Memory Management:**
- Intelligent caching system with configurable limits
- NNImage cache: up to 200,000 color mappings
- RepliKate cache: up to 20,000 tensor sequences  
- Automatic memory monitoring and garbage collection optimization

### Quality Control Parameters

**Processing Thresholds:**
- Edge threshold: 0.01 (controls RepliKate usage - lower = more AI processing)
- Smoothness threshold: 0.001 (controls bilinear interpolation usage)
- Progressive step size: 1.25x per increment (balance of quality vs speed)

**Method Distribution (typical):**
- ~40% Bilinear interpolation (smooth areas)
- ~50% NNImage multi-scale processing (moderate detail)
- ~10% RepliKate sequence prediction (high detail edges)

### Supported Formats & Features

**Image Formats:**
- Input/Output: PNG, JPEG, BMP, TIFF, WebP
- Color depth: 8-bit and 16-bit per channel
- Full alpha channel support with proper blending
- Maintains original color profiles and metadata where possible

**Upscaling Modes:**
- Progressive upscaling: 1.25x → 1.5x → 2x → 4x (and beyond)
- Single-pass upscaling for smaller scale factors
- 1x cleanup mode (enhancement without scaling)
- Specialized pixel art mode with palette preservation

**Enhancement Features:**
- 5-pass hyper-detailing pipeline with artifact detection
- Edge-preserving smoothing and adaptive sharpening
- JPEG artifact detection and removal
- Genuine detail preservation vs compression artifact removal

### System Requirements

**Minimum:**
- .NET 9.0 runtime
- 4GB RAM (8GB+ recommended for large images)
- Multi-core CPU (4+ cores recommended)

**Recommended:**
- 16GB+ RAM for processing large images
- NVIDIA GPU with CUDA support (GTX 1060 or better)
- NVMe SSD for faster image I/O
- 8+ CPU cores for maximum parallel processing efficiency

**Optimal Performance:**
- High-core-count CPU (12+ cores) 
- Modern NVIDIA GPU (RTX series recommended)
- 32GB+ RAM for batch processing
- Fast storage (NVMe SSD) for large image files
