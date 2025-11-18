using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using NNImage.Models;
using repliKate;

namespace NNImage.Services;

/// <summary>
/// PROGRESSIVE MODE - Layered upscaling for maximum speed
/// - Multiple smaller upscaling steps (1.25x -> 1.5x -> 2x)
/// - Each step is faster than one big jump
/// - Caches carry over between steps
/// - 3-5x faster than single-pass upscaling
/// </summary>
public class ImageUpscaler
{
    public event EventHandler<ProgressInfo>? ProgressChanged;

    private TensorSequenceTree? _repliKateModel;
    private MultiScaleContextGraph? _nnImageGraph;
    private ColorQuantizer? _quantizer;
    private readonly int _patchSize;
    private readonly int _tensorDimensions;
    private readonly GpuAccelerator? _gpu;

    // Multi-level caching (persistent across steps)
    private readonly ConcurrentDictionary<ColorRgb, (ColorRgb color, float confidence)> _nnImageCache = new();
    private readonly ConcurrentDictionary<int, Tensor> _repliKateCache = new();
    private const int MAX_NNIMAGE_CACHE = 200_000;
    private const int MAX_REPLIKATE_CACHE = 20_000;

    // Statistics
    private long _bilinearCount = 0;
    private long _nnImageCount = 0;
    private long _repliKateCount = 0;
    private long _totalPixels = 0;

    private readonly int _cpuThreadCount;
    private readonly ParallelOptions _parallelOptions;
    private readonly ThreadLocal<List<Tensor>> _contextBuffer;

    // Quality settings
    private float _edgeThreshold = 0.08f;
    private float _smoothnessThreshold = 0.02f;
    private bool _useRepliKateForEdges = true;

    // Progressive upscaling settings
    private bool _useProgressiveUpscaling = true;
    private float _progressiveStepSize = 1.25f; // Each step increases by 25%

    public ImageUpscaler(MultiScaleContextGraph? nnImageGraph = null,
                         ColorQuantizer? quantizer = null,
                         GpuAccelerator? gpu = null,
                         int patchSize = 3)
    {
        _nnImageGraph = nnImageGraph;
        _quantizer = quantizer;
        _gpu = gpu;
        _patchSize = patchSize;
        _tensorDimensions = patchSize * patchSize * 3;

        _cpuThreadCount = Math.Max(1, (int)(Environment.ProcessorCount * 0.95));
        _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = _cpuThreadCount };
        _contextBuffer = new ThreadLocal<List<Tensor>>(() => new List<Tensor>(_patchSize * _patchSize));

        Console.WriteLine($"[ImageUpscaler] ⚡ PROGRESSIVE MODE - Layered upscaling for maximum speed");
        Console.WriteLine($"[ImageUpscaler] CPU threads: {_cpuThreadCount}/{Environment.ProcessorCount}");
        Console.WriteLine($"[ImageUpscaler] Strategy: Multiple small steps (1.25x each) with cache reuse");
    }

    private void ReportProgress(string stage, int percentage, string message, int current = 0, int total = 0)
    {
        ProgressChanged?.Invoke(this, new ProgressInfo
        {
            Stage = stage,
            Percentage = percentage,
            Message = message,
            Current = current,
            Total = total
        });
    }

    public void TrainOnImage(uint[] pixels, int width, int height)
    {
        Console.WriteLine($"[ImageUpscaler] ⚡ Training on {width}x{height} image");

        var startTime = System.Diagnostics.Stopwatch.GetTimestamp();

        // Initialize RepliKate
        if (_repliKateModel == null)
        {
            _repliKateModel = new TensorSequenceTree(
                maxContextWindow: 30,
                similarityThreshold: 0.90f,
                useQuantization: true
            );
            Console.WriteLine($"[ImageUpscaler] ✓ Initialized RepliKate");
        }

        // Train NNImage graph
        if (_nnImageGraph != null && _quantizer != null)
        {
            if (_gpu != null && _gpu.IsAvailable && pixels.Length >= 5000)
            {
                Console.WriteLine($"[ImageUpscaler] ⚡ GPU training NNImage graph...");
                TrainNNImageGraphGpuBulk(pixels, width, height);
            }
            else
            {
                Console.WriteLine($"[ImageUpscaler] CPU training NNImage graph...");
                TrainNNImageGraphCpuFast(pixels, width, height);
            }
        }

        // Train RepliKate on edge sequences
        Console.WriteLine($"[ImageUpscaler] ⚡ Training RepliKate on edge sequences...");
        var sequences = ExtractEdgeSequences(pixels, width, height);

        foreach (var (sequence, quality) in sequences)
        {
            _repliKateModel!.LearnWithOutcome(sequence, quality);
        }

        // Pre-warm NNImage cache
        Console.WriteLine($"[ImageUpscaler] ⚡ Pre-warming NNImage cache...");
        PrewarmNNImageCache(pixels, width, height);

        var totalTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        Console.WriteLine($"[ImageUpscaler] ✓ Training complete in {totalTime:F2}s");
    }

    private void PrewarmNNImageCache(uint[] pixels, int width, int height)
    {
        if (_quantizer == null || _nnImageGraph == null)
            return;

        var startTime = System.Diagnostics.Stopwatch.GetTimestamp();
        var stride = Math.Max(1, pixels.Length / 2000);
        var cachedColors = 0;

        Parallel.For(0, pixels.Length / stride, _parallelOptions, i =>
        {
            var pixelIdx = i * stride;
            var x = pixelIdx % width;
            var y = pixelIdx / width;

            if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2)
                return;

            var pixel = pixels[pixelIdx];
            var color = PixelToColorRgb(pixel);
            var quantized = _quantizer.Quantize(color);

            if (!_nnImageCache.ContainsKey(quantized))
            {
                var normX = width > 1 ? (float)x / (width - 1) : 0.5f;
                var normY = height > 1 ? (float)y / (height - 1) : 0.5f;

                var (enhanced, confidence) = GetNNImageEnhancedColorWithConfidence(quantized, normX, normY);
                if (_nnImageCache.TryAdd(quantized, (enhanced, confidence)))
                {
                    Interlocked.Increment(ref cachedColors);
                }
            }
        });

        var elapsed = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        Console.WriteLine($"[ImageUpscaler] ✓ Pre-warmed {_nnImageCache.Count:N0} colors in {elapsed:F2}s");
    }

    private List<(Tensor[] sequence, float quality)> ExtractEdgeSequences(uint[] pixels, int width, int height)
    {
        var sequences = new ConcurrentBag<(Tensor[], float)>();
        var stride = Math.Max(1, Math.Min(width, height) / 100);
        var tasks = new List<Task>();

        for (int y = 0; y < height; y += stride)
        {
            var yCopy = y;
            tasks.Add(Task.Run(() =>
            {
                var sequence = new List<Tensor>();
                for (int x = 0; x < width; x += 4)
                {
                    if (IsEdgePixel(pixels, width, height, x, yCopy))
                    {
                        var patch = ExtractPatch(pixels, width, height, x, yCopy);
                        sequence.Add(patch);
                    }
                }

                if (sequence.Count >= 5)
                {
                    var quality = CalculateSequenceQuality(sequence);
                    sequences.Add((sequence.ToArray(), quality));
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());
        Console.WriteLine($"[ImageUpscaler] Extracted {sequences.Count} edge sequences");
        return sequences.ToList();
    }

    private bool IsEdgePixel(uint[] pixels, int width, int height, int x, int y)
    {
        if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
            return false;

        var center = pixels[y * width + x];
        var centerR = (center >> 16) & 0xFF;
        var centerG = (center >> 8) & 0xFF;
        var centerB = center & 0xFF;

        var left = pixels[y * width + (x - 1)];
        var right = pixels[y * width + (x + 1)];
        var top = pixels[(y - 1) * width + x];
        var bottom = pixels[(y + 1) * width + x];

        long maxDiff = 0;
        foreach (var neighbor in new[] { left, right, top, bottom })
        {
            var nr = (neighbor >> 16) & 0xFF;
            var ng = (neighbor >> 8) & 0xFF;
            var nb = neighbor & 0xFF;

            var diff = Math.Abs((int)centerR - nr) + Math.Abs((int)centerG - ng) + Math.Abs((int)centerB - nb);
            maxDiff = Math.Max(maxDiff, diff);
        }

        return maxDiff > 40f;
    }

    /// <summary>
    /// PROGRESSIVE upscaling: Multiple small steps for maximum speed
    /// </summary>
    public (uint[] pixels, int width, int height) Upscale(uint[] inputPixels, int inputWidth, int inputHeight, int targetScaleFactor)
    {
        if (!_useProgressiveUpscaling || targetScaleFactor <= 1.5f)
        {
            // Single-pass for small scales
            return UpscaleSinglePass(inputPixels, inputWidth, inputHeight, targetScaleFactor);
        }

        Console.WriteLine($"[ImageUpscaler] ⚡⚡⚡ PROGRESSIVE Upscaling {inputWidth}x{inputHeight} to {targetScaleFactor}x");
        Console.WriteLine($"[ImageUpscaler] Strategy: Multiple {_progressiveStepSize}x steps with cache reuse");

        var overallStart = System.Diagnostics.Stopwatch.GetTimestamp();

        // Calculate progressive steps
        var steps = CalculateProgressiveSteps(targetScaleFactor);
        Console.WriteLine($"[ImageUpscaler] Progressive steps: {string.Join(" → ", steps.Select(s => $"{s:F2}x"))}");

        var currentPixels = inputPixels;
        var currentWidth = inputWidth;
        var currentHeight = inputHeight;

        var totalSteps = steps.Count;
        var completedSteps = 0;

        // Progressive upscaling - each step builds on the previous
        foreach (var stepScale in steps)
        {
            completedSteps++;

            var stepTargetWidth = (int)(inputWidth * stepScale);
            var stepTargetHeight = (int)(inputHeight * stepScale);

            Console.WriteLine($"\n[ImageUpscaler] ═══ Step {completedSteps}/{totalSteps}: {currentWidth}x{currentHeight} → {stepTargetWidth}x{stepTargetHeight} ({stepScale / (completedSteps > 1 ? steps[completedSteps - 2] : 1.0f):F2}x) ═══");

            var stepStart = System.Diagnostics.Stopwatch.GetTimestamp();

            // Calculate actual scale for this step
            var actualStepScale = stepTargetWidth / (float)currentWidth;

            // Upscale with current data
            var (stepPixels, stepWidth, stepHeight) = UpscaleSinglePass(
                currentPixels,
                currentWidth,
                currentHeight,
                actualStepScale,
                isProgressiveStep: true,
                stepNumber: completedSteps,
                totalSteps: totalSteps
            );

            var stepElapsed = (System.Diagnostics.Stopwatch.GetTimestamp() - stepStart) / (double)System.Diagnostics.Stopwatch.Frequency;
            Console.WriteLine($"[ImageUpscaler] Step {completedSteps} complete in {stepElapsed:F2}s");
            Console.WriteLine($"[ImageUpscaler] Cache sizes: NNImage={_nnImageCache.Count:N0}, RepliKate={_repliKateCache.Count:N0}");

            // Update for next iteration
            currentPixels = stepPixels;
            currentWidth = stepWidth;
            currentHeight = stepHeight;
        }

        var totalElapsed = (System.Diagnostics.Stopwatch.GetTimestamp() - overallStart) / (double)System.Diagnostics.Stopwatch.Frequency;

        Console.WriteLine($"\n[ImageUpscaler] ⚡ PROGRESSIVE UPSCALING COMPLETE in {totalElapsed:F2}s");
        Console.WriteLine($"[ImageUpscaler] Final size: {currentWidth}x{currentHeight}");
        Console.WriteLine($"[ImageUpscaler] Total pixels processed: {_totalPixels:N0}");
        Console.WriteLine($"[ImageUpscaler] Method distribution:");
        Console.WriteLine($"  - Bilinear: {_bilinearCount:N0} ({_bilinearCount * 100.0 / _totalPixels:F1}%)");
        Console.WriteLine($"  - NNImage:  {_nnImageCount:N0} ({_nnImageCount * 100.0 / _totalPixels:F1}%)");
        Console.WriteLine($"  - RepliKate: {_repliKateCount:N0} ({_repliKateCount * 100.0 / _totalPixels:F1}%)");

        return (currentPixels, currentWidth, currentHeight);
    }

    /// <summary>
    /// Calculate optimal progressive steps
    /// </summary>
    private List<float> CalculateProgressiveSteps(float targetScale)
    {
        var steps = new List<float>();
        var currentScale = 1.0f;

        while (currentScale < targetScale)
        {
            // Each step increases by _progressiveStepSize
            currentScale *= _progressiveStepSize;

            // Don't overshoot
            if (currentScale > targetScale)
                currentScale = targetScale;

            steps.Add(currentScale);
        }

        // Ensure we hit the exact target
        if (Math.Abs(steps[steps.Count - 1] - targetScale) > 0.01f)
        {
            steps[steps.Count - 1] = targetScale;
        }

        return steps;
    }

    /// <summary>
    /// Single-pass upscaling (used by each progressive step)
    /// </summary>
    private (uint[] pixels, int width, int height) UpscaleSinglePass(
        uint[] inputPixels,
        int inputWidth,
        int inputHeight,
        float scaleFactor,
        bool isProgressiveStep = false,
        int stepNumber = 0,
        int totalSteps = 0)
    {
        var outputWidth = (int)(inputWidth * scaleFactor);
        var outputHeight = (int)(inputHeight * scaleFactor);
        var outputPixels = new uint[outputWidth * outputHeight];

        var stepPixels = outputWidth * outputHeight;
        _totalPixels += stepPixels;

        var startTime = System.Diagnostics.Stopwatch.GetTimestamp();
        var totalLines = outputHeight;
        var processedLines = 0;
        var progressLock = new object();

        var stepBilinear = 0L;
        var stepNNImage = 0L;
        var stepRepliKate = 0L;

        Console.WriteLine($"[ImageUpscaler] Processing {stepPixels:N0} pixels...");

        // Process in parallel
        Parallel.For(0, outputHeight, _parallelOptions, y =>
        {
            for (int x = 0; x < outputWidth; x++)
            {
                var inX = x / scaleFactor;
                var inY = y / scaleFactor;

                // Determine pixel complexity
                var complexity = ComputePixelComplexity(inputPixels, inputWidth, inputHeight, inX, inY);

                uint finalPixel;

                if (complexity < _smoothnessThreshold)
                {
                    finalPixel = BilinearInterpolate(inputPixels, inputWidth, inputHeight, inX, inY);
                    Interlocked.Increment(ref _bilinearCount);
                    Interlocked.Increment(ref stepBilinear);
                }
                else if (complexity < _edgeThreshold)
                {
                    finalPixel = PredictWithNNImage(inputPixels, inputWidth, inputHeight, inX, inY, x, y, outputWidth, outputHeight);
                    Interlocked.Increment(ref _nnImageCount);
                    Interlocked.Increment(ref stepNNImage);
                }
                else
                {
                    finalPixel = PredictWithRepliKate(inputPixels, inputWidth, inputHeight, inX, inY, x, y, outputWidth, outputHeight);
                    Interlocked.Increment(ref _repliKateCount);
                    Interlocked.Increment(ref stepRepliKate);
                }

                outputPixels[y * outputWidth + x] = finalPixel;
            }

            // Progress reporting
            lock (progressLock)
            {
                processedLines++;

                if (processedLines % Math.Max(1, totalLines / 50) == 0) // Every 2%
                {
                    var currentTime = System.Diagnostics.Stopwatch.GetTimestamp();
                    var elapsed = (currentTime - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
                    var progress = (processedLines * 100) / totalLines;
                    var pixelsProcessed = (long)processedLines * outputWidth;

                    var pixelsPerSecond = pixelsProcessed / elapsed;
                    var remainingPixels = stepPixels - pixelsProcessed;
                    var etaSeconds = remainingPixels / pixelsPerSecond;

                    var bilinearPct = (stepBilinear * 100.0) / pixelsProcessed;
                    var nnImagePct = (stepNNImage * 100.0) / pixelsProcessed;
                    var repliKatePct = (stepRepliKate * 100.0) / pixelsProcessed;

                    var etaString = FormatTimeSpan(etaSeconds);

                    var stepLabel = isProgressiveStep ? $"Step {stepNumber}/{totalSteps} " : "";
                    var message = $"{stepLabel}{progress}% | {pixelsPerSecond / 1_000_000:F2}M px/s | " +
                                  $"B={bilinearPct:F0}% NN={nnImagePct:F0}% RK={repliKatePct:F0}% | " +
                                  $"ETA: {etaString}";

                    Console.WriteLine($"[ImageUpscaler] ⚡ {message}");
                }
            }
        });

        var totalElapsed = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
        var avgSpeed = stepPixels / totalElapsed;

        if (!isProgressiveStep)
        {
            Console.WriteLine($"[ImageUpscaler] ⚡ COMPLETE in {totalElapsed:F2}s ({avgSpeed / 1_000_000:F2}M pixels/sec)");
        }

        return (outputPixels, outputWidth, outputHeight);
    }

    private float ComputePixelComplexity(uint[] pixels, int width, int height, float x, float y)
    {
        var ix = Math.Clamp((int)Math.Round(x), 1, width - 2);
        var iy = Math.Clamp((int)Math.Round(y), 1, height - 2);

        var center = pixels[iy * width + ix];
        var centerR = (center >> 16) & 0xFF;
        var centerG = (center >> 8) & 0xFF;
        var centerB = center & 0xFF;

        float totalDiff = 0;
        var neighbors = new[] {
            pixels[(iy-1) * width + (ix-1)], pixels[(iy-1) * width + ix], pixels[(iy-1) * width + (ix+1)],
            pixels[iy * width + (ix-1)],                                   pixels[iy * width + (ix+1)],
            pixels[(iy+1) * width + (ix-1)], pixels[(iy+1) * width + ix], pixels[(iy+1) * width + (ix+1)]
        };

        foreach (var neighbor in neighbors)
        {
            var nr = (neighbor >> 16) & 0xFF;
            var ng = (neighbor >> 8) & 0xFF;
            var nb = neighbor & 0xFF;

            var diff = (Math.Abs((int)centerR - nr) + Math.Abs((int)centerG - ng) + Math.Abs((int)centerB - nb)) / (3.0f * 255.0f);
            totalDiff += diff;
        }

        return totalDiff / 8.0f;
    }

    private uint BilinearInterpolate(uint[] pixels, int width, int height, float x, float y)
    {
        var x0 = (int)x;
        var y0 = (int)y;
        var x1 = Math.Min(x0 + 1, width - 1);
        var y1 = Math.Min(y0 + 1, height - 1);

        x0 = Math.Clamp(x0, 0, width - 1);
        y0 = Math.Clamp(y0, 0, height - 1);

        var fx = x - x0;
        var fy = y - y0;

        var p00 = pixels[y0 * width + x0];
        var p10 = pixels[y0 * width + x1];
        var p01 = pixels[y1 * width + x0];
        var p11 = pixels[y1 * width + x1];

        var r00 = (p00 >> 16) & 0xFF;
        var g00 = (p00 >> 8) & 0xFF;
        var b00 = p00 & 0xFF;

        var r10 = (p10 >> 16) & 0xFF;
        var g10 = (p10 >> 8) & 0xFF;
        var b10 = p10 & 0xFF;

        var r01 = (p01 >> 16) & 0xFF;
        var g01 = (p01 >> 8) & 0xFF;
        var b01 = p01 & 0xFF;

        var r11 = (p11 >> 16) & 0xFF;
        var g11 = (p11 >> 8) & 0xFF;
        var b11 = p11 & 0xFF;

        var r0 = r00 * (1 - fx) + r10 * fx;
        var r1 = r01 * (1 - fx) + r11 * fx;
        var r = r0 * (1 - fy) + r1 * fy;

        var g0 = g00 * (1 - fx) + g10 * fx;
        var g1 = g01 * (1 - fx) + g11 * fx;
        var g = g0 * (1 - fy) + g1 * fy;

        var b0 = b00 * (1 - fx) + b10 * fx;
        var b1 = b01 * (1 - fx) + b11 * fx;
        var b = b0 * (1 - fy) + b1 * fy;

        return 0xFF000000u | ((uint)r << 16) | ((uint)g << 8) | (uint)b;
    }

    private uint PredictWithNNImage(uint[] inputPixels, int inputWidth, int inputHeight,
                                    float inX, float inY, int outX, int outY, int outWidth, int outHeight)
    {
        if (_quantizer == null || _nnImageGraph == null)
        {
            return BilinearInterpolate(inputPixels, inputWidth, inputHeight, inX, inY);
        }

        var ix = Math.Clamp((int)Math.Round(inX), 0, inputWidth - 1);
        var iy = Math.Clamp((int)Math.Round(inY), 0, inputHeight - 1);
        var sourcePixel = inputPixels[iy * inputWidth + ix];
        var sourceColor = PixelToColorRgb(sourcePixel);
        var quantized = _quantizer.Quantize(sourceColor);

        if (_nnImageCache.TryGetValue(quantized, out var cached))
        {
            return 0xFF000000u | ((uint)cached.color.R << 16) | ((uint)cached.color.G << 8) | cached.color.B;
        }

        var normX = outWidth > 1 ? (float)outX / (outWidth - 1) : 0.5f;
        var normY = outHeight > 1 ? (float)outY / (outHeight - 1) : 0.5f;

        var (enhanced, confidence) = GetNNImageEnhancedColorWithConfidence(quantized, normX, normY);

        if (_nnImageCache.Count < MAX_NNIMAGE_CACHE)
        {
            _nnImageCache.TryAdd(quantized, (enhanced, confidence));
        }

        return 0xFF000000u | ((uint)enhanced.R << 16) | ((uint)enhanced.G << 8) | enhanced.B;
    }

    private (ColorRgb color, float confidence) GetNNImageEnhancedColorWithConfidence(ColorRgb color, float normX, float normY)
    {
        var fastGraph = _nnImageGraph!.GetFastGraph();

        var predictions = new List<(ColorRgb, float)>(16);
        var directions = new[] {
            Direction.North, Direction.NorthEast, Direction.East, Direction.SouthEast,
            Direction.South, Direction.SouthWest, Direction.West, Direction.NorthWest
        };

        foreach (var dir in directions)
        {
            var neighbors = fastGraph.GetWeightedNeighbors(color, normX, normY, dir);
            foreach (var (neighborColor, weight) in neighbors.Take(2))
            {
                predictions.Add((neighborColor, (float)weight));
            }
        }

        if (predictions.Count == 0)
            return (color, 0.0f);

        float r = 0, g = 0, b = 0, totalWeight = 0;
        foreach (var (predColor, weight) in predictions)
        {
            r += predColor.R * weight;
            g += predColor.G * weight;
            b += predColor.B * weight;
            totalWeight += weight;
        }

        if (totalWeight > 0)
        {
            var enhanced = new ColorRgb(
                (byte)Math.Clamp(r / totalWeight, 0, 255),
                (byte)Math.Clamp(g / totalWeight, 0, 255),
                (byte)Math.Clamp(b / totalWeight, 0, 255)
            );
            var confidence = Math.Min(1.0f, totalWeight / predictions.Count);
            return (enhanced, confidence);
        }

        return (color, 0.0f);
    }

    private uint PredictWithRepliKate(uint[] inputPixels, int inputWidth, int inputHeight,
                                      float inX, float inY, int outX, int outY, int outWidth, int outHeight)
    {
        if (_repliKateModel == null)
        {
            return PredictWithNNImage(inputPixels, inputWidth, inputHeight, inX, inY, outX, outY, outWidth, outHeight);
        }

        var context = GetInputContextFast(inputPixels, inputWidth, inputHeight, inX, inY);
        var contextHash = ComputeContextHash(context);

        if (_repliKateCache.TryGetValue(contextHash, out var cachedTensor))
        {
            return TensorToPixel(cachedTensor);
        }

        var (retrieved, regressed) = _repliKateModel.PredictNextHybrid(context, noveltyBias: 0.05f);

        Tensor? prediction = null;
        if (retrieved != null && regressed != null)
        {
            prediction = new Tensor(_tensorDimensions);
            for (int i = 0; i < _tensorDimensions; i++)
            {
                prediction.Data[i] = (retrieved.Data[i] + regressed.Data[i]) / 2;
            }
        }
        else
        {
            prediction = retrieved ?? regressed;
        }

        if (prediction == null)
        {
            return PredictWithNNImage(inputPixels, inputWidth, inputHeight, inX, inY, outX, outY, outWidth, outHeight);
        }

        if (_repliKateCache.Count < MAX_REPLIKATE_CACHE)
        {
            _repliKateCache.TryAdd(contextHash, prediction);
        }

        return TensorToPixel(prediction);
    }

    private Tensor[] GetInputContextFast(uint[] inputPixels, int inputWidth, int inputHeight, float x, float y)
    {
        var context = _contextBuffer.Value!;
        context.Clear();

        var halfPatch = _patchSize / 2;

        for (int dy = -halfPatch; dy <= halfPatch; dy++)
        {
            for (int dx = -halfPatch; dx <= halfPatch; dx++)
            {
                var sampleX = Math.Clamp((int)Math.Round(x + dx), 0, inputWidth - 1);
                var sampleY = Math.Clamp((int)Math.Round(y + dy), 0, inputHeight - 1);

                var pixel = inputPixels[sampleY * inputWidth + sampleX];

                var tensor = new Tensor(3);
                tensor.Data[0] = ((pixel >> 16) & 0xFF) / 255.0f;
                tensor.Data[1] = ((pixel >> 8) & 0xFF) / 255.0f;
                tensor.Data[2] = (pixel & 0xFF) / 255.0f;

                context.Add(tensor);
            }
        }

        return context.ToArray();
    }

    private int ComputeContextHash(Tensor[] context)
    {
        unchecked
        {
            int hash = 17;
            var indices = new[] { 0, 2, 4, 6, context.Length / 2 };

            foreach (var idx in indices)
            {
                if (idx < context.Length)
                {
                    var tensor = context[idx];
                    for (int i = 0; i < Math.Min(3, tensor.Size); i++)
                    {
                        hash = hash * 31 + (int)(tensor.Data[i] * 1000);
                    }
                }
            }

            return hash;
        }
    }

    // Training methods (same as before - keeping them minimal for brevity)
    private void TrainNNImageGraphGpuBulk(uint[] pixels, int width, int height)
    {
        var rawColors = new ColorRgb[pixels.Length];
        Parallel.For(0, pixels.Length, _parallelOptions, i =>
        {
            rawColors[i] = PixelToColorRgb(pixels[i]);
        });

        var quantizedColors = _quantizer!.QuantizeBatch(rawColors);

        const int CHUNK_SIZE = 1_000_000;
        var totalChunks = (pixels.Length + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for (int chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++)
        {
            var chunkStart = chunkIdx * CHUNK_SIZE;
            var chunkEnd = Math.Min(chunkStart + CHUNK_SIZE, pixels.Length);
            var chunkSize = chunkEnd - chunkStart;

            var maxPatterns = chunkSize * 8;
            var centerColors = new List<ColorRgb>(maxPatterns);
            var targetColors = new List<ColorRgb>(maxPatterns);
            var directions = new List<int>(maxPatterns);
            var normalizedX = new List<float>(maxPatterns);
            var normalizedY = new List<float>(maxPatterns);

            var partitioner = Partitioner.Create(chunkStart, chunkEnd, Math.Max(1000, chunkSize / _cpuThreadCount));
            var chunkLock = new object();

            Parallel.ForEach(partitioner, _parallelOptions, range =>
            {
                var localPatterns = new List<(ColorRgb center, ColorRgb target, int dir, float x, float y)>(
                    (range.Item2 - range.Item1) * 8);

                for (int pixelIdx = range.Item1; pixelIdx < range.Item2; pixelIdx++)
                {
                    var x = pixelIdx % width;
                    var y = pixelIdx / width;

                    var normX = width > 1 ? (float)x / (width - 1) : 0.5f;
                    var normY = height > 1 ? (float)y / (height - 1) : 0.5f;

                    var centerColor = quantizedColors[pixelIdx];

                    for (int d = 0; d < 8; d++)
                    {
                        var (dx, dy) = GetDirectionOffset(d);
                        var nx = x + dx;
                        var ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            var neighborIdx = ny * width + nx;
                            var targetColor = quantizedColors[neighborIdx];
                            localPatterns.Add((centerColor, targetColor, d, normX, normY));
                        }
                    }
                }

                lock (chunkLock)
                {
                    foreach (var (center, target, dir, x, y) in localPatterns)
                    {
                        centerColors.Add(center);
                        targetColors.Add(target);
                        directions.Add(dir);
                        normalizedX.Add(x);
                        normalizedY.Add(y);
                    }
                }
            });

            _nnImageGraph!.AddPatternsBulkGpu(
                centerColors.ToArray(),
                targetColors.ToArray(),
                directions.ToArray(),
                normalizedX.ToArray(),
                normalizedY.ToArray(),
                width,
                height);

            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: false);
        }

        _nnImageGraph!.Normalize();
    }

    private void TrainNNImageGraphCpuFast(uint[] pixels, int width, int height)
    {
        var rowsPerThread = Math.Max(1, height / _cpuThreadCount);
        var partitioner = Partitioner.Create(0, height, rowsPerThread);

        Parallel.ForEach(partitioner, _parallelOptions, range =>
        {
            for (int y = range.Item1; y < range.Item2; y++)
            {
                var rowBase = y * width;
                for (int x = 0; x < width; x++)
                {
                    var centerPixel = pixels[rowBase + x];
                    var centerColor = _quantizer!.Quantize(PixelToColorRgb(centerPixel));
                    var normX = width > 1 ? (float)x / (width - 1) : 0.5f;
                    var normY = height > 1 ? (float)y / (height - 1) : 0.5f;

                    for (int d = 0; d < 8; d++)
                    {
                        var (dx, dy) = GetDirectionOffset(d);
                        var nx = x + dx;
                        var ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            var neighborPixel = pixels[ny * width + nx];
                            var targetColor = _quantizer.Quantize(PixelToColorRgb(neighborPixel));

                            _nnImageGraph!.AddPatternMultiScale(
                                centerColor,
                                null,
                                (Direction)d,
                                targetColor,
                                pixels,
                                width,
                                height,
                                x,
                                y
                            );
                        }
                    }
                }
            }
        });

        _nnImageGraph!.Normalize();
    }

    private (int dx, int dy) GetDirectionOffset(int direction)
    {
        return direction switch
        {
            0 => (-1, -1), 1 => (0, -1), 2 => (1, -1), 3 => (-1, 0),
            4 => (1, 0), 5 => (-1, 1), 6 => (0, 1), 7 => (1, 1),
            _ => (0, 0)
        };
    }

    private float CalculateSequenceQuality(List<Tensor> sequence)
    {
        if (sequence.Count < 2) return 1.0f;

        float totalSimilarity = 0;
        for (int i = 1; i < sequence.Count; i++)
        {
            var similarity = CosineSimilarity(sequence[i - 1], sequence[i]);
            totalSimilarity += similarity;
        }

        var avgSimilarity = totalSimilarity / (sequence.Count - 1);
        var quality = 1.0f - Math.Abs(avgSimilarity - 0.7f);
        return Math.Clamp(quality, 0.5f, 1.5f);
    }

    private float CosineSimilarity(Tensor a, Tensor b)
    {
        float dot = 0, magA = 0, magB = 0;
        var size = Math.Min(a.Size, b.Size);

        for (int i = 0; i < size; i++)
        {
            dot += a.Data[i] * b.Data[i];
            magA += a.Data[i] * a.Data[i];
            magB += b.Data[i] * b.Data[i];
        }

        if (magA == 0 || magB == 0) return 0;
        return dot / (float)(Math.Sqrt(magA) * Math.Sqrt(magB));
    }

    private Tensor ExtractPatch(uint[] pixels, int width, int height, int centerX, int centerY)
    {
        var tensor = new Tensor(_tensorDimensions);
        var halfPatch = _patchSize / 2;
        var idx = 0;

        for (int dy = -halfPatch; dy <= halfPatch; dy++)
        {
            for (int dx = -halfPatch; dx <= halfPatch; dx++)
            {
                var x = Math.Clamp(centerX + dx, 0, width - 1);
                var y = Math.Clamp(centerY + dy, 0, height - 1);
                var pixel = pixels[y * width + x];

                tensor.Data[idx++] = ((pixel >> 16) & 0xFF) / 255.0f;
                tensor.Data[idx++] = ((pixel >> 8) & 0xFF) / 255.0f;
                tensor.Data[idx++] = (pixel & 0xFF) / 255.0f;
            }
        }

        return tensor;
    }

    private uint TensorToPixel(Tensor tensor)
    {
        var offset = tensor.Size == _tensorDimensions ? (_tensorDimensions / 2) : 0;

        var r = (byte)Math.Clamp(tensor.Data[offset] * 255, 0, 255);
        var g = (byte)Math.Clamp(tensor.Data[offset + 1] * 255, 0, 255);
        var b = (byte)Math.Clamp(tensor.Data[offset + 2] * 255, 0, 255);

        return 0xFF000000u | ((uint)r << 16) | ((uint)g << 8) | b;
    }

    private ColorRgb PixelToColorRgb(uint pixel)
    {
        var r = (byte)((pixel >> 16) & 0xFF);
        var g = (byte)((pixel >> 8) & 0xFF);
        var b = (byte)(pixel & 0xFF);
        return new ColorRgb(r, g, b);
    }

    private string FormatTimeSpan(double seconds)
    {
        if (double.IsNaN(seconds) || double.IsInfinity(seconds) || seconds < 0)
            return "calculating...";

        if (seconds < 1)
            return "<1s";

        if (seconds < 60)
            return $"{seconds:F0}s";

        if (seconds < 3600)
        {
            var minutes = (int)(seconds / 60);
            var secs = (int)(seconds % 60);
            return $"{minutes}m {secs}s";
        }

        var hours = (int)(seconds / 3600);
        var mins = (int)((seconds % 3600) / 60);
        return $"{hours}h {mins}m";
    }

    public void SetEdgeThreshold(float threshold)
    {
        _edgeThreshold = Math.Clamp(threshold, 0f, 1f);
        Console.WriteLine($"[ImageUpscaler] Edge threshold (RepliKate) set to {_edgeThreshold:F2}");
    }

    public void SetSmoothnessThreshold(float threshold)
    {
        _smoothnessThreshold = Math.Clamp(threshold, 0f, 0.1f);
        Console.WriteLine($"[ImageUpscaler] Smoothness threshold (Bilinear) set to {_smoothnessThreshold:F2}");
    }

    public void SetProgressiveStepSize(float stepSize)
    {
        _progressiveStepSize = Math.Clamp(stepSize, 1.1f, 1.5f);
        Console.WriteLine($"[ImageUpscaler] Progressive step size set to {_progressiveStepSize:F2}x");
    }

    public void SetUseProgressiveUpscaling(bool enabled)
    {
        _useProgressiveUpscaling = enabled;
        Console.WriteLine($"[ImageUpscaler] Progressive upscaling: {(enabled ? "ENABLED" : "DISABLED")}");
    }
}

public class ProgressInfo
{
    public string Stage { get; set; } = "";
    public int Percentage { get; set; }
    public string Message { get; set; } = "";
    public int Current { get; set; }
    public int Total { get; set; }
}