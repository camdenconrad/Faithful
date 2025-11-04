using System;
using System.Linq;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using NNImage.Models;

namespace NNImage.Services;

public class GpuAccelerator : IDisposable
{
    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private bool _disposed;

    public bool IsAvailable { get; private set; }

    public GpuAccelerator()
    {
        try
        {
            Console.WriteLine("[GPU] Initializing CUDA accelerator...");
            _context = Context.Create(builder => builder.Cuda().EnableAlgorithms());

            // Try to get CUDA accelerator
            var cudaDevice = _context.Devices.OfType<CudaDevice>().FirstOrDefault();

            if (cudaDevice != null)
            {
                _accelerator = cudaDevice.CreateAccelerator(_context);
                IsAvailable = true;
                Console.WriteLine($"[GPU] CUDA accelerator initialized: {_accelerator.Name}");
                Console.WriteLine($"[GPU] Memory: {_accelerator.MemorySize / (1024 * 1024)} MB");
                Console.WriteLine($"[GPU] Max threads per group: {_accelerator.MaxNumThreadsPerGroup}");
            }
            else
            {
                Console.WriteLine("[GPU] No CUDA device found, using CPU fallback");
                _accelerator = _context.GetPreferredDevice(false).CreateAccelerator(_context);
                IsAvailable = false;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GPU] Failed to initialize: {ex.Message}");
            IsAvailable = false;

            // Fallback to CPU
            _context = Context.Create(builder => builder.CPU().EnableAlgorithms());
            _accelerator = _context.GetPreferredDevice(false).CreateAccelerator(_context);
        }
    }

    // GPU Kernel: Calculate distance from color to centroid
    private static void CalculateDistancesKernel(
        Index1D index,
        ArrayView<byte> colorsR,
        ArrayView<byte> colorsG,
        ArrayView<byte> colorsB,
        ArrayView<byte> centroidsR,
        ArrayView<byte> centroidsG,
        ArrayView<byte> centroidsB,
        ArrayView<int> assignments,
        int numCentroids)
    {
        int colorIdx = index;
        byte r = colorsR[colorIdx];
        byte g = colorsG[colorIdx];
        byte b = colorsB[colorIdx];

        float minDistance = float.MaxValue;
        int bestCentroid = 0;

        for (int c = 0; c < numCentroids; c++)
        {
            int dr = r - centroidsR[c];
            int dg = g - centroidsG[c];
            int db = b - centroidsB[c];
            float distance = dr * dr + dg * dg + db * db;

            if (distance < minDistance)
            {
                minDistance = distance;
                bestCentroid = c;
            }
        }

        assignments[colorIdx] = bestCentroid;
    }

    public int[] AssignColorsToNearestCentroid(ColorRgb[] colors, ColorRgb[] centroids)
    {
        if (!IsAvailable || colors.Length < 10000) // Use GPU only for large datasets
        {
            return AssignColorsToNearestCentroidCpu(colors, centroids);
        }

        try
        {
            // Separate RGB channels for GPU
            var colorsR = colors.Select(c => c.R).ToArray();
            var colorsG = colors.Select(c => c.G).ToArray();
            var colorsB = colors.Select(c => c.B).ToArray();

            var centroidsR = centroids.Select(c => c.R).ToArray();
            var centroidsG = centroids.Select(c => c.G).ToArray();
            var centroidsB = centroids.Select(c => c.B).ToArray();

            // Allocate GPU memory
            using var deviceColorsR = _accelerator.Allocate1D<byte>(colorsR.Length);
            using var deviceColorsG = _accelerator.Allocate1D<byte>(colorsG.Length);
            using var deviceColorsB = _accelerator.Allocate1D<byte>(colorsB.Length);
            using var deviceCentroidsR = _accelerator.Allocate1D<byte>(centroidsR.Length);
            using var deviceCentroidsG = _accelerator.Allocate1D<byte>(centroidsG.Length);
            using var deviceCentroidsB = _accelerator.Allocate1D<byte>(centroidsB.Length);
            using var deviceAssignments = _accelerator.Allocate1D<int>(colors.Length);

            // Copy to GPU
            deviceColorsR.CopyFromCPU(colorsR);
            deviceColorsG.CopyFromCPU(colorsG);
            deviceColorsB.CopyFromCPU(colorsB);
            deviceCentroidsR.CopyFromCPU(centroidsR);
            deviceCentroidsG.CopyFromCPU(centroidsG);
            deviceCentroidsB.CopyFromCPU(centroidsB);

            // Load and execute kernel
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
                ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
                ArrayView<int>, int>(CalculateDistancesKernel);

            kernel((int)deviceColorsR.Length, 
                deviceColorsR.View, deviceColorsG.View, deviceColorsB.View,
                deviceCentroidsR.View, deviceCentroidsG.View, deviceCentroidsB.View,
                deviceAssignments.View, centroids.Length);

            // Wait for completion
            _accelerator.Synchronize();

            // Copy results back
            var assignments = deviceAssignments.GetAsArray1D();

            Console.WriteLine($"[GPU] Assigned {colors.Length} colors to {centroids.Length} centroids");
            return assignments;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GPU] Kernel execution failed, falling back to CPU: {ex.Message}");
            return AssignColorsToNearestCentroidCpu(colors, centroids);
        }
    }

    private int[] AssignColorsToNearestCentroidCpu(ColorRgb[] colors, ColorRgb[] centroids)
    {
        var assignments = new int[colors.Length];

        System.Threading.Tasks.Parallel.For(0, colors.Length, i =>
        {
            var color = colors[i];
            var minDistance = float.MaxValue;
            var bestCentroid = 0;

            for (int c = 0; c < centroids.Length; c++)
            {
                var centroid = centroids[c];
                var dr = color.R - centroid.R;
                var dg = color.G - centroid.G;
                var db = color.B - centroid.B;
                var distance = dr * dr + dg * dg + db * db;

                if (distance < minDistance)
                {
                    minDistance = distance;
                    bestCentroid = c;
                }
            }

            assignments[i] = bestCentroid;
        });

        return assignments;
    }

    // GPU Kernel: Extract adjacency patterns
    private static void ExtractAdjacenciesKernel(
        Index1D index,
        ArrayView<uint> pixels,
        ArrayView<byte> paletteR,
        ArrayView<byte> paletteG,
        ArrayView<byte> paletteB,
        ArrayView<int> adjacencyBuffer,
        int width,
        int height,
        int paletteSize)
    {
        int pixelIndex = index;
        int x = pixelIndex % width;
        int y = pixelIndex / width;

        if (x >= width || y >= height) return;

        uint centerPixel = pixels[pixelIndex];
        byte centerR = (byte)((centerPixel >> 16) & 0xFF);
        byte centerG = (byte)((centerPixel >> 8) & 0xFF);
        byte centerB = (byte)(centerPixel & 0xFF);

        // Find nearest palette color for center
        int centerColor = FindNearestPaletteColor(centerR, centerG, centerB, 
            paletteR, paletteG, paletteB, paletteSize);

        // Check 8 directions
        int[] dx = { 0, 1, 1, 1, 0, -1, -1, -1 };
        int[] dy = { -1, -1, 0, 1, 1, 1, 0, -1 };

        for (int dir = 0; dir < 8; dir++)
        {
            int nx = x + dx[dir];
            int ny = y + dy[dir];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                int neighborIndex = ny * width + nx;
                uint neighborPixel = pixels[neighborIndex];
                byte neighborR = (byte)((neighborPixel >> 16) & 0xFF);
                byte neighborG = (byte)((neighborPixel >> 8) & 0xFF);
                byte neighborB = (byte)(neighborPixel & 0xFF);

                int neighborColor = FindNearestPaletteColor(neighborR, neighborG, neighborB,
                    paletteR, paletteG, paletteB, paletteSize);

                // Store adjacency: centerColor * 8 * paletteSize + dir * paletteSize + neighborColor
                int bufferIndex = centerColor * 8 * paletteSize + dir * paletteSize + neighborColor;
                Atomic.Add(ref adjacencyBuffer[bufferIndex], 1);
            }
        }
    }

    private static int FindNearestPaletteColor(
        byte r, byte g, byte b,
        ArrayView<byte> paletteR,
        ArrayView<byte> paletteG,
        ArrayView<byte> paletteB,
        int paletteSize)
    {
        int minDistance = int.MaxValue;
        int bestIndex = 0;

        for (int i = 0; i < paletteSize; i++)
        {
            int dr = r - paletteR[i];
            int dg = g - paletteG[i];
            int db = b - paletteB[i];
            int distance = dr * dr + dg * dg + db * db;

            if (distance < minDistance)
            {
                minDistance = distance;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    public int[] ExtractAdjacenciesGpu(uint[] pixels, int width, int height, ColorRgb[] palette)
    {
        if (!IsAvailable || pixels.Length < 50000) // Use GPU only for large images
        {
            return null; // Signal to use CPU fallback
        }

        try
        {
            Console.WriteLine($"[GPU] Extracting adjacencies for {width}x{height} image with {palette.Length} colors");

            var paletteR = palette.Select(c => c.R).ToArray();
            var paletteG = palette.Select(c => c.G).ToArray();
            var paletteB = palette.Select(c => c.B).ToArray();

            // Allocate GPU memory
            using var devicePixels = _accelerator.Allocate1D<uint>(pixels.Length);
            using var devicePaletteR = _accelerator.Allocate1D<byte>(paletteR.Length);
            using var devicePaletteG = _accelerator.Allocate1D<byte>(paletteG.Length);
            using var devicePaletteB = _accelerator.Allocate1D<byte>(paletteB.Length);

            // Buffer size: palette_size * 8_directions * palette_size
            int bufferSize = palette.Length * 8 * palette.Length;
            using var deviceAdjacencyBuffer = _accelerator.Allocate1D<int>(bufferSize);

            // Copy to GPU
            devicePixels.CopyFromCPU(pixels);
            devicePaletteR.CopyFromCPU(paletteR);
            devicePaletteG.CopyFromCPU(paletteG);
            devicePaletteB.CopyFromCPU(paletteB);
            deviceAdjacencyBuffer.MemSetToZero();

            // Load and execute kernel
            var kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<uint>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
                ArrayView<int>, int, int, int>(ExtractAdjacenciesKernel);

            kernel((int)devicePixels.Length,
                devicePixels.View,
                devicePaletteR.View, devicePaletteG.View, devicePaletteB.View,
                deviceAdjacencyBuffer.View,
                width, height, palette.Length);

            // Wait for completion
            _accelerator.Synchronize();

            // Copy results back
            var adjacencyBuffer = deviceAdjacencyBuffer.GetAsArray1D();

            Console.WriteLine($"[GPU] Extracted adjacencies successfully");
            return adjacencyBuffer;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GPU] Adjacency extraction failed: {ex.Message}");
            return null; // Signal to use CPU fallback
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Console.WriteLine("[GPU] Disposing GPU accelerator");
            _accelerator?.Dispose();
            _context?.Dispose();
            _disposed = true;
        }
    }
}
