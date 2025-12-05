[# Faithful - Pixel Art Upscaling Done Right

**Preserving Every Pixel, Every Color, Every Detail — Exactly As Intended**

When you upscale pixel art, you don't want AI inventing details that were never there. Faithful is built on one core principle: **be faithful to the input image**. No hallucinated details, no invented textures, no AI "creativity" — just your original artwork, beautifully scaled up.

## Perfect for Pixel Art & Game Assets

Faithful excels at upscaling pixel art, retro game textures, and low-resolution artwork where **preserving the original is critical**:

- **100% Faithful Reproduction**: Never adds details that weren't in the original
- **Palette Preservation**: Maintains exact color palettes from pixel art
- **Grid-Based Filtering**: Intelligently detects and preserves pixel art structure
- **Zero Smearing**: Unlike other upscalers, keeps sharp edges crisp and clean

### Before & After Examples

*Examples coming soon - pixel art sprites, game textures, and retro graphics*

## Why Faithful Beats the Competition

| Input Type |  Faithful | ESRGAN / SwinIR / etc. | Winner        |
|------------|-------------------------|------------------------|---------------|
| **Pixel art** | 100% perfect, zero smearing | Melts it into watercolor | **Faithful** |
| **Hand-painted game textures** | Preserves every intended stroke | Adds fake pores, fake grit | **Faithful**       |
| **256-512px classic 3D textures** | Looks like the artist painted it at 4K | Hallucinates new details | **Faithful**       |
| **PS1 / N64 / early 2000s assets** | Restores original intent flawlessly | Turns blocky polygons into plastic skin | **Faithful**       |
| **Sprites & icons** | Crisp edges, perfect colors | Blurry mess with color bleed | **Faithful**       |
| **Actual low-res photographs** | Faithful to source (no fake detail) | Invents plausible but fake detail | **ESRGAN**    |

### The Faithfulness Philosophy

Faithful does **one thing exceptionally well**: it faithfully upscales your image without adding details that weren't there.

**What "Faithful" Means:**
- ✅ Preserves exact color palettes from the original
- ✅ Maintains pixel grid structure in pixel art
- ✅ Keeps edges sharp and clean, never blurred
- ✅ Respects the artist's original intent
- ❌ Never invents textures or details
- ❌ Never hallucinates features
- ❌ Never "improves" with AI creativity

#### Comparison Gallery

Can't tell the difference? It's because Faithful is doing its job.

Before

<img width="240" height="150" alt="khyzha-074460" src="https://github.com/user-attachments/assets/e066d8fc-9fc0-4725-865a-23e99b1ff4e6" />
After

<img width="1200" height="750" alt="upscaled_20251205_023356" src="https://github.com/user-attachments/assets/b673156a-114a-4c32-86f3-07cff68d18fd" />

Before

<img width="300" height="300" alt="remember-that-there-are-things-you-cant-control-809245" src="https://github.com/user-attachments/assets/d19dd715-083d-4e83-a0b9-fa6b53629e90" />

After

<img width="900" height="900" alt="upscaled_20251205_023306" src="https://github.com/user-attachments/assets/b5bec9ab-da82-4555-b5de-7b79ccce4326" />

Before

<img width="240" height="150" alt="khyzha-074460" src="https://github.com/user-attachments/assets/a79d5ab7-b034-429d-abe2-7eb6db6e14d6" />


After (Experimental Faithful++ detail addition)

<img width="1920" height="1200" alt="upscaled_20251205_023005" src="https://github.com/user-attachments/assets/03a743d3-48a7-4d1b-a6a0-549c73d402e5" />



## What This Actually Does

Faithful combines several different AI approaches to upscale images with absolute faithfulness to the original. Unlike most upscalers that either blur everything or invent fake details, this one analyzes what the original artist actually created and scales it up without adding or removing anything.

## Main Features

### Pixel Art Mode - The Star Feature

Faithful includes a dedicated **Pixel Art Mode** specifically designed for upscaling retro game graphics, sprites, and low-resolution artwork. Unlike generic upscalers that blur or hallucinate details, Pixel Art Mode is **obsessively faithful** to your original image.

**Key Features:**
- **Intelligent Density Detection**: Automatically analyzes your original image to detect pixel grid size (1x1, 2x2, 4x4, 8x8)
- **Grid-Based Majority Filtering**: Each detected grid cell becomes its most common color, eliminating AI-induced blur
- **Original Palette Enforcement**: Only uses colors from your source image — no color invention
- **Nearest-Neighbor Pre-Scaling**: Small images get crisp pre-scaling before AI refinement
- **Reduced Sharpening**: Gentle enhancement preserves clean pixel art aesthetic

#### Pixel Art Mode Results

<!-- Add pixel art comparison image here: Original | Other AI | Faithful -->

**How It Works:**
1. Extracts exact color palette from original image (typically 16-256 colors)
2. Analyzes original to detect pixel density (NOT the blurry upscaled version)
3. Applies intelligent grid-based filtering at detected density scale
4. Ensures every output pixel uses only original palette colors
5. Result: Perfectly faithful upscale that looks hand-painted at higher resolution

### Progressive Upscaling
Instead of jumping straight from 1x to 4x (which usually looks terrible), this does multiple smaller steps: 1x → 1.25x → 1.5x → 2x → 4x. Each step builds on the last one.

There's also a "1x mode" that just cleans up the image without making it bigger - great for fixing compression artifacts and general cleanup.

### The Three AI Systems (All Trained on YOUR Image)

**Multi-Scale Context Graphs**: This is the main workhorse. It learns color relationships and spatial patterns **exclusively from your input image**, then applies them intelligently during upscaling. No pre-trained models, no generic datasets — it only knows what you show it. GPU-accelerated and analyzes 8 different directions of pattern recognition.

**Sequence Prediction (RepliKate)**: This one specializes in edges and fine details by learning from **your image's own sequences**. It studies horizontal, vertical, and diagonal patterns in your artwork to predict how pixels should look. Perfect for text, sharp edges, and intricate details that other systems mess up. Learns **only** from your input.

**Intelligent Routing**: The system automatically decides which method to use for each pixel:
- About 40% gets basic bilinear interpolation (for smooth areas)
- About 50% gets Multi-Scale Context processing (moderate detail)  
- About 10% gets RepliKate sequence prediction (high detail edges and textures)

**Why This Matters:** Because the AI trains **only on your image**, it can never invent details or hallucinate features. It can only reproduce patterns it learned from your original artwork.

### Artifact Detection and Handling

The system can tell the difference between real image detail and compression artifacts. It identifies JPEG compression blocks, analyzes multi-scale consistency, and protects genuine textures while removing junk.

When hyper-detailing is enabled, it runs a 5-pass enhancement:
1. Conservative GPU enhancement on genuine detail areas only
2. Faithful micro-details for structured edges
3. Lighter GPU refinement pass
4. Adaptive sharpening with artifact protection  
5. Surgical smoothing that only removes confirmed artifacts

## Image Generation Capabilities

### Color Adjacency Neural Generator

Faithful includes a sophisticated image generation system that learns spatial color relationships from training images and creates entirely new images using Wave Function Collapse algorithms.

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
- Faithful cache: up to 200,000 color mappings
- Faithful cache: up to 20,000 tensor sequences  
- Automatic memory monitoring and garbage collection optimization

### Quality Control Parameters

**Processing Thresholds:**
- Edge threshold: 0.01 (controls Faithful usage - lower = more AI processing)
- Smoothness threshold: 0.001 (controls bilinear interpolation usage)
- Progressive step size: 1.25x per increment (balance of quality vs speed)

**Method Distribution (typical):**
- ~40% Bilinear interpolation (smooth areas)
- ~50% Faithful multi-scale processing (moderate detail)
- ~10% Faithful (repliKate) sequence prediction (high detail edges)

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
**Optimal Performance:**
- High-core-count CPU (12+ cores) 
- Modern NVIDIA GPU (RTX 40xx series recommended)
- 4GB RAM
- Fast storage (NVMe SSD) for large image files
