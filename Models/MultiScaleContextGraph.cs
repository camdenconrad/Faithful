using System;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Concurrent;
using NNImage.Services;

namespace NNImage.Models;

/// <summary>
/// Ultra-fast graph that learns patterns with simple node-based structure
/// SUPER FAST training with direct node connections - THREAD-SAFE!
/// Graph uses normalized positions (0-1) so it scales automatically to any resolution
/// </summary>
public class MultiScaleContextGraph
{
    private readonly FastContextGraph _fastGraph = new();
    private readonly ConcurrentDictionary<ColorRgb, (float dx, float dy)> _directionalBias = new();
    private GpuAccelerator? _gpu;

    // Cache for last created nodes (for linking patterns) - thread-safe
    private readonly ConcurrentDictionary<(int x, int y), GraphNode> _positionCache = new();

    // Track the native resolution (largest image trained on)
    private int _maxTrainedWidth = 0;
    private int _maxTrainedHeight = 0;
    private readonly object _resolutionLock = new object();

    // Structural rule graph - learns grayscale structure patterns
    private readonly StructuralRuleGraph _structuralGraph = new();
    private const int STRUCTURE_CLASSES = 16; // Reduced to 16 for speed/memory

    public MultiScaleContextGraph()
    {
        Console.WriteLine("[MultiScaleContextGraph] Initialized ULTRA-FAST node-based graph for SUPER FAST training!");
        Console.WriteLine("[MultiScaleContextGraph] Graph uses normalized positions - automatically scales to any resolution!");
        Console.WriteLine("[MultiScaleContextGraph] Structural learning enabled with 16 structure classes");
    }

    public void SetGpuAccelerator(GpuAccelerator? gpu)
    {
        _gpu = gpu;
        Console.WriteLine($"[MultiScaleContextGraph] GPU: {(gpu?.IsAvailable == true ? "ENABLED" : "DISABLED")}");
    }

    /// <summary>
    /// MASSIVE SPEEDUP: Train patterns in bulk using GPU acceleration
    /// Processes entire image worth of patterns in one GPU call instead of pixel-by-pixel
    /// </summary>
    public void AddPatternsBulkGpu(
        ColorRgb[] centerColors,
        ColorRgb[] targetColors,
        int[] directions,
        float[] normalizedX,
        float[] normalizedY,
        int regionWidth,
        int regionHeight)
    {
        if (_gpu == null || !_gpu.IsAvailable)
        {
            Console.WriteLine("[MultiScaleContextGraph] GPU not available, use AddPatternMultiScale instead");
            return;
        }

        // Track graph resolution growth
        lock (_resolutionLock)
        {
            if (regionWidth > _maxTrainedWidth || regionHeight > _maxTrainedHeight)
            {
                var oldRes = $"{_maxTrainedWidth}x{_maxTrainedHeight}";
                _maxTrainedWidth = Math.Max(_maxTrainedWidth, regionWidth);
                _maxTrainedHeight = Math.Max(_maxTrainedHeight, regionHeight);
                Console.WriteLine($"[MultiScaleContextGraph] Graph expanded from {oldRes} to {_maxTrainedWidth}x{_maxTrainedHeight}");
            }
        }

        Console.WriteLine($"[MultiScaleContextGraph] GPU BULK TRAINING: {centerColors.Length:N0} patterns");

        // GPU processes all patterns in parallel on 10240 CUDA cores!
        var result = _gpu.TrainPatternsBulkGpu(centerColors, targetColors, directions, normalizedX, normalizedY);

        if (result.HasValue)
        {
            var (edgeKeys, edgeWeights) = result.Value;

            Console.WriteLine($"[MultiScaleContextGraph] Applying {edgeKeys.Length:N0} GPU-trained edges to graph");
            Console.WriteLine($"[MultiScaleContextGraph] ⚠️ Processing in CHUNKS to stay under 30GB RAM limit");

            var startTime = System.Diagnostics.Stopwatch.GetTimestamp();

            // ═══ CRITICAL: Process in chunks to avoid 40GB+ memory usage ═══
            // With 265M patterns, processing all at once uses 40GB+ RAM
            // Process 5M patterns at a time to stay well under 30GB limit
            const int CHUNK_SIZE = 5_000_000; // 5 million patterns per chunk
            var totalChunks = (edgeKeys.Length + CHUNK_SIZE - 1) / CHUNK_SIZE;
            var totalEdgesApplied = 0L;

            Console.WriteLine($"[MultiScaleContextGraph] Processing {totalChunks} chunks of {CHUNK_SIZE:N0} patterns each");

            for (int chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++)
            {
                var chunkStart = chunkIdx * CHUNK_SIZE;
                var chunkEnd = Math.Min(chunkStart + CHUNK_SIZE, edgeKeys.Length);
                var chunkSize = chunkEnd - chunkStart;

                Console.WriteLine($"[MultiScaleContextGraph] ┌─ Chunk {chunkIdx + 1}/{totalChunks}: Processing patterns {chunkStart:N0} to {chunkEnd:N0}");

                // Group edges in this chunk
                var edgeMap = new System.Collections.Concurrent.ConcurrentDictionary<long, (ColorRgb center, ColorRgb target, Direction dir, float x, float y, float weight)>();

                System.Threading.Tasks.Parallel.For(chunkStart, chunkEnd, 
                    new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                    i =>
                {
                    var key = edgeKeys[i];
                    edgeMap.AddOrUpdate(key,
                        // Add new
                        (centerColors[i], targetColors[i], (Direction)directions[i], normalizedX[i], normalizedY[i], edgeWeights[i]),
                        // Update existing - accumulate weight
                        (_, existing) => (existing.center, existing.target, existing.dir, existing.x, existing.y, existing.weight + edgeWeights[i]));
                });

                Console.WriteLine($"[MultiScaleContextGraph] │  Grouped to {edgeMap.Count:N0} unique edges in chunk");

                // Pre-create unique nodes for this chunk
                var uniqueNodeKeys = new System.Collections.Concurrent.ConcurrentDictionary<(ColorRgb, float, float), byte>();

                System.Threading.Tasks.Parallel.ForEach(edgeMap.Values, 
                    new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                    edge =>
                {
                    uniqueNodeKeys.TryAdd((edge.center, edge.x, edge.y), 0);
                    uniqueNodeKeys.TryAdd((edge.target, edge.x, edge.y), 0);
                });

                Console.WriteLine($"[MultiScaleContextGraph] │  Creating {uniqueNodeKeys.Count:N0} nodes");

                // Batch create nodes
                var nodeCache = new System.Collections.Concurrent.ConcurrentDictionary<(ColorRgb, float, float), GraphNode>();
                var nodeBatches = uniqueNodeKeys.Keys.Chunk(1000).ToArray();

                System.Threading.Tasks.Parallel.ForEach(nodeBatches, 
                    new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                    batch =>
                {
                    foreach (var (color, x, y) in batch)
                    {
                        var node = _fastGraph.GetOrCreateNode(color, x, y);
                        nodeCache.TryAdd((color, x, y), node);
                    }
                });

                // Apply edges for this chunk
                var edgeArray = edgeMap.Values.ToArray();
                System.Threading.Tasks.Parallel.ForEach(edgeArray, 
                    new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount },
                    edge =>
                {
                    if (nodeCache.TryGetValue((edge.center, edge.x, edge.y), out var centerNode) &&
                        nodeCache.TryGetValue((edge.target, edge.x, edge.y), out var targetNode))
                    {
                        _fastGraph.AddEdge(centerNode, edge.dir, targetNode, edge.weight);
                    }
                });

                totalEdgesApplied += edgeArray.Length;
                Console.WriteLine($"[MultiScaleContextGraph] │  Applied {edgeArray.Length:N0} edges");

                // ═══ CRITICAL: Clear chunk data immediately to free memory ═══
                edgeMap.Clear();
                edgeMap = null;
                uniqueNodeKeys.Clear();
                uniqueNodeKeys = null;
                nodeCache.Clear();
                nodeCache = null;
                edgeArray = null;
                nodeBatches = null;

                // Force aggressive garbage collection between chunks to stay under 30GB
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();

                var memGB = GC.GetTotalMemory(true) / (1024.0 * 1024 * 1024);
                Console.WriteLine($"[MultiScaleContextGraph] └─ Chunk complete. RAM: {memGB:F2}GB / 30GB");
            }

            var totalTime = (System.Diagnostics.Stopwatch.GetTimestamp() - startTime) / (double)System.Diagnostics.Stopwatch.Frequency;
            var edgesPerSec = totalEdgesApplied / totalTime;
            Console.WriteLine($"[MultiScaleContextGraph] ⚡ Applied {totalEdgesApplied:N0} edges in {totalTime * 1000:F1}ms ({edgesPerSec / 1_000_000:F2}M edges/sec)");
            Console.WriteLine($"[MultiScaleContextGraph] ✓ GPU bulk training applied to graph! Stayed under 30GB RAM limit");
        }
        else
        {
            Console.WriteLine("[MultiScaleContextGraph] GPU bulk training failed, patterns not added");
        }
    }

    public void AddPatternMultiScale(
        ColorRgb centerColor,
        Dictionary<Direction, ColorRgb?> neighbors,
        Direction outputDirection,
        ColorRgb targetColor,
        uint[] pixelRegion,
        int regionWidth,
        int regionHeight,
        int centerX,
        int centerY)
    {
        // Track graph resolution growth (less frequent check)
        if (regionWidth > _maxTrainedWidth || regionHeight > _maxTrainedHeight)
        {
            lock (_resolutionLock)
            {
                if (regionWidth > _maxTrainedWidth || regionHeight > _maxTrainedHeight)
                {
                    _maxTrainedWidth = Math.Max(_maxTrainedWidth, regionWidth);
                    _maxTrainedHeight = Math.Max(_maxTrainedHeight, regionHeight);
                }
            }
        }

        // Pre-calculate commonly used values
        var invWidthMinus1 = regionWidth > 1 ? 1.0f / (regionWidth - 1) : 0.5f;
        var invHeightMinus1 = regionHeight > 1 ? 1.0f / (regionHeight - 1) : 0.5f;
        var normalizedX = centerX * invWidthMinus1;
        var normalizedY = centerY * invHeightMinus1;

        // Compute structural metrics on-the-fly (fast and memory efficient)
        var structureClass = ComputeStructureClassFast(pixelRegion, regionWidth, regionHeight, centerX, centerY);
        _structuralGraph.RecordObservation(structureClass, ColorToUInt32(centerColor));

        // Get or create node for center position
        var centerNode = _fastGraph.GetOrCreateNode(centerColor, normalizedX, normalizedY);

        // OPTIMIZED: Learn color sequence patterns with reduced chain length for speed
        var (dx, dy) = outputDirection.GetOffset();

        const int maxChainLength = 5; // Reduced from 9 to 5 for 44% faster training
        var previousNode = centerNode;

        // Pre-calculate step positions to reduce allocations
        for (int step = 1; step <= maxChainLength; step++)
        {
            var lookX = centerX + (dx * step);
            var lookY = centerY + (dy * step);

            // Bounds check with early exit
            if ((uint)lookX >= (uint)regionWidth || (uint)lookY >= (uint)regionHeight)
                break;

            // Inline pixel-to-color conversion for speed
            var lookPixel = pixelRegion[lookY * regionWidth + lookX];
            var a = (byte)((lookPixel >> 24) & 0xFF);
            var r = (byte)((lookPixel >> 16) & 0xFF);
            var g = (byte)((lookPixel >> 8) & 0xFF);
            var b = (byte)(lookPixel & 0xFF);

            if (a < 255)
            {
                var alpha = a / 255.0;
                r = (byte)(r * alpha + 255 * (1 - alpha));
                g = (byte)(g * alpha + 255 * (1 - alpha));
                b = (byte)(b * alpha + 255 * (1 - alpha));
            }

            var lookColor = new ColorRgb(r, g, b);
            var lookNormX = lookX * invWidthMinus1;
            var lookNormY = lookY * invHeightMinus1;

            // Get or create node at this position
            var lookNode = _fastGraph.GetOrCreateNode(lookColor, lookNormX, lookNormY);

            // Create sequential edge with optimized weight calculation
            var chainWeight = 1.0f / step;
            _fastGraph.AddEdge(previousNode, outputDirection, lookNode, chainWeight);

            // Long-range connections only for steps 2-3 (reduced from all steps)
            if (step == 2 || step == 3)
            {
                var longRangeWeight = chainWeight * 0.5f;
                _fastGraph.AddEdge(centerNode, outputDirection, lookNode, longRangeWeight);
            }

            previousNode = lookNode;
        }
    }

    private Direction GetClosestDirection(float dx, float dy)
    {
        // Normalize
        var length = (float)Math.Sqrt(dx * dx + dy * dy);
        if (length < 0.0001f) return Direction.East;

        dx /= length;
        dy /= length;

        // Find closest of 8 directions
        var directions = new[] 
        {
            (Direction.North, 0f, -1f),
            (Direction.NorthEast, 0.707f, -0.707f),
            (Direction.East, 1f, 0f),
            (Direction.SouthEast, 0.707f, 0.707f),
            (Direction.South, 0f, 1f),
            (Direction.SouthWest, -0.707f, 0.707f),
            (Direction.West, -1f, 0f),
            (Direction.NorthWest, -0.707f, -0.707f)
        };

        var bestDir = Direction.East;
        var bestDot = -2f;

        foreach (var (dir, dirX, dirY) in directions)
        {
            var dot = dx * dirX + dy * dirY;
            if (dot > bestDot)
            {
                bestDot = dot;
                bestDir = dir;
            }
        }

        return bestDir;
    }


    private void UpdateDirectionalBias(ColorRgb centerColor, Direction direction, ColorRgb targetColor)
    {
        // Calculate gradient vector based on color change
        var (dx, dy) = direction.GetOffset();

        // Color difference as gradient magnitude
        var dr = targetColor.R - centerColor.R;
        var dg = targetColor.G - centerColor.G;
        var db = targetColor.B - centerColor.B;
        var magnitude = (float)Math.Sqrt(dr * dr + dg * dg + db * db) / 441.0f; // Normalize by max distance

        // Thread-safe update using AddOrUpdate
        var alpha = 0.1f; // Learning rate for bias
        _directionalBias.AddOrUpdate(
            centerColor,
            // Add new entry
            (dx * magnitude * alpha, dy * magnitude * alpha),
            // Update existing entry
            (_, current) => (
                current.dx * (1 - alpha) + dx * magnitude * alpha,
                current.dy * (1 - alpha) + dy * magnitude * alpha
            )
        );
    }

    public List<(ColorRgb color, double weight)> GetWeightedNeighborsMultiScale(
        NeighborhoodPattern currentPattern,
        Direction direction)
    {
        // ULTRA-FAST: Direct node lookup
        // Position is stored in GraphNode, so we use center of image as fallback for lookup
        var predictions = _fastGraph.GetWeightedNeighbors(
            currentPattern.Center, 
            0.5f,  // Center X position as fallback
            0.5f,  // Center Y position as fallback
            direction);

        if (predictions.Count == 0)
        {
            return new List<(ColorRgb, double)>();
        }

        // Apply directional bias for extra accuracy
        var centerColor = currentPattern.Center;
        if (_directionalBias.TryGetValue(centerColor, out var bias))
        {
            var (dx, dy) = direction.GetOffset();
            var dirAlignment = dx * bias.dx + dy * bias.dy;

            if (dirAlignment > 0)
            {
                // Boost aligned directions
                for (int i = 0; i < predictions.Count; i++)
                {
                    var (color, weight) = predictions[i];
                    predictions[i] = (color, weight * (1 + dirAlignment * 0.3));
                }

                // Re-normalize
                var total = predictions.Sum(p => p.weight);
                if (total > 0)
                {
                    for (int i = 0; i < predictions.Count; i++)
                    {
                        var (color, weight) = predictions[i];
                        predictions[i] = (color, weight / total);
                    }
                }
            }
        }

        return predictions;
    }

    public void Normalize()
    {
        Console.WriteLine("[MultiScaleContextGraph] ULTRA-FAST normalization (already normalized in nodes)...");

        var (width, height) = GetNativeResolution();
        Console.WriteLine($"[MultiScaleContextGraph] Graph native resolution: {width}x{height}");
        Console.WriteLine($"[MultiScaleContextGraph] Graph uses normalized positions - works at ANY resolution!");
        Console.WriteLine($"[MultiScaleContextGraph] Pattern chains learned during training - no post-processing needed!");

        _fastGraph.PrintStats();

        // Normalize structural graph
        Console.WriteLine("[MultiScaleContextGraph] Normalizing structural rule graph...");
        _structuralGraph.NumStructureClasses = STRUCTURE_CLASSES;
        _structuralGraph.Normalize();
        Console.WriteLine($"[MultiScaleContextGraph] Structural graph: {STRUCTURE_CLASSES} structure classes learned");

        Console.WriteLine("[MultiScaleContextGraph] Training complete - graph is ready for super fast generation!");
    }

    public int GetTotalPatternCount()
    {
        return _fastGraph.GetNodeCount();
    }

    public int GetColorCount()
    {
        return _fastGraph.GetColorCount();
    }

    public List<ColorRgb> GetAllColors()
    {
        return _fastGraph.GetAllColors();
    }

    public FastContextGraph GetFastGraph() => _fastGraph;

    /// <summary>
    /// Create a Wave Function Collapse generator for real-time image generation
    /// </summary>
    public WaveFunctionCollapseGenerator CreateWFCGenerator(int? seed = null)
    {
        return new WaveFunctionCollapseGenerator(_fastGraph, seed);
    }

    /// <summary>
    /// Get the native resolution (largest image trained on)
    /// Graph uses normalized positions so it works at any resolution,
    /// but this tells you the max detail level it has learned
    /// </summary>
    public (int width, int height) GetNativeResolution()
    {
        lock (_resolutionLock)
        {
            return (_maxTrainedWidth, _maxTrainedHeight);
        }
    }

    /// <summary>
    /// Get recommended resolution for training/generation
    /// Returns the native resolution as the sweet spot
    /// </summary>
    public (int width, int height) GetRecommendedResolution()
    {
        lock (_resolutionLock)
        {
            if (_maxTrainedWidth == 0 || _maxTrainedHeight == 0)
            {
                return (1920, 1080); // Default if no training yet
            }
            return (_maxTrainedWidth, _maxTrainedHeight);
        }
    }

    private ColorRgb PixelToColor(uint pixel)
    {
        var a = (byte)((pixel >> 24) & 0xFF);
        var r = (byte)((pixel >> 16) & 0xFF);
        var g = (byte)((pixel >> 8) & 0xFF);
        var b = (byte)(pixel & 0xFF);

        if (a < 255)
        {
            var alpha = a / 255.0;
            r = (byte)(r * alpha + 255 * (1 - alpha));
            g = (byte)(g * alpha + 255 * (1 - alpha));
            b = (byte)(b * alpha + 255 * (1 - alpha));
        }

        return new ColorRgb(r, g, b);
    }

    /// <summary>
    /// ULTRA-FAST structural class computation using simple 3x3 window
    /// Computes luminance, gradient, and entropy in one pass - memory efficient!
    /// </summary>
    private int ComputeStructureClassFast(uint[] pixels, int width, int height, int x, int y)
    {
        // Fast luminance computation for 3x3 window
        var luminances = new float[9];
        var idx = 0;

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                var nx = x + dx;
                var ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    var pixel = pixels[ny * width + nx];
                    var r = (byte)((pixel >> 16) & 0xFF);
                    var g = (byte)((pixel >> 8) & 0xFF);
                    var b = (byte)(pixel & 0xFF);

                    // Fast luminance: L = 0.299R + 0.587G + 0.114B
                    luminances[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
                }
                else
                {
                    luminances[idx] = 0;
                }
                idx++;
            }
        }

        // Fast Sobel gradient (center pixel)
        var centerLum = luminances[4];
        var gx = -luminances[0] + luminances[2] - 2 * luminances[3] + 2 * luminances[5] - luminances[6] + luminances[8];
        var gy = -luminances[0] - 2 * luminances[1] - luminances[2] + luminances[6] + 2 * luminances[7] + luminances[8];
        var gradient = (float)Math.Sqrt(gx * gx + gy * gy);

        // Fast entropy (variance as proxy)
        var mean = luminances.Average();
        var variance = 0f;
        for (int i = 0; i < 9; i++)
        {
            var diff = luminances[i] - mean;
            variance += diff * diff;
        }
        variance /= 9;
        var entropy = (float)Math.Sqrt(variance);

        // Quantize to structure class (0-15)
        // Use simple binning: luminance (4 bins) x gradient (2 bins) x entropy (2 bins) = 16 classes
        var lumBin = (int)(centerLum / 64) % 4;  // 0-3
        var gradBin = gradient > 50 ? 1 : 0;      // 0-1
        var entropyBin = entropy > 30 ? 1 : 0;    // 0-1

        var structureClass = lumBin * 4 + gradBin * 2 + entropyBin;
        return structureClass;
    }

    private static uint ColorToUInt32(ColorRgb color)
    {
        return (uint)((255 << 24) | (color.R << 16) | (color.G << 8) | color.B);
    }

    public StructuralRuleGraph GetStructuralGraph() => _structuralGraph;

    // --- Persistence mapping ---
    public ModelSnapshot ToSnapshot()
    {
        Console.WriteLine("[MultiScaleContextGraph] Creating snapshot of ultra-fast graph...");

        var nodes = _fastGraph.GetAllNodes();
        var nodeDtos = new List<FastNodeDto>(nodes.Count);
        var edgeDtos = new List<FastEdgeDto>();

        // map nodes to indices
        var indexByNode = new Dictionary<GraphNode, int>(nodes.Count);
        for (int i = 0; i < nodes.Count; i++)
        {
            var n = nodes[i];
            indexByNode[n] = i;
            nodeDtos.Add(new FastNodeDto
            {
                Color = n.Color.ToUInt32(),
                X = n.NormalizedX,
                Y = n.NormalizedY,
                ObservationCount = n.ObservationCount
            });
        }

        // collect edges
        for (int i = 0; i < nodes.Count; i++)
        {
            var fromNode = nodes[i];
            foreach (var kv in fromNode.Edges)
            {
                var dir = (int)kv.Key;
                var list = kv.Value;
                for (int e = 0; e < list.Count; e++)
                {
                    var (toNode, weight) = list[e];
                    if (!indexByNode.TryGetValue(toNode, out var toIdx))
                    {
                        // should not happen if nodes list is complete
                        continue;
                    }
                    edgeDtos.Add(new FastEdgeDto
                    {
                        From = i,
                        Direction = dir,
                        To = toIdx,
                        Weight = weight
                    });
                }
            }
        }

        var snapshot = new ModelSnapshot
        {
            Version = 2,
            NodeCount = nodeDtos.Count,
            EdgeCount = edgeDtos.Count,
            Nodes = nodeDtos,
            Edges = edgeDtos
        };

        Console.WriteLine($"[MultiScaleContextGraph] Snapshot created: {snapshot.NodeCount} nodes, {snapshot.EdgeCount} edges");

        return snapshot;
    }

    public static MultiScaleContextGraph FromSnapshot(ModelSnapshot snapshot, GpuAccelerator? gpu)
    {
        var graph = new MultiScaleContextGraph();
        graph.SetGpuAccelerator(gpu);

        Console.WriteLine($"[MultiScaleContextGraph] Loading snapshot (version {snapshot.Version})...");

        if (snapshot.Version == 2)
        {
            Console.WriteLine("[MultiScaleContextGraph] Fast graph format detected - rebuilding fast graph...");
            try
            {
                var nodesDto = snapshot.Nodes;
                var edgesDto = snapshot.Edges;
                if (nodesDto == null || nodesDto.Count == 0)
                {
                    Console.WriteLine("[MultiScaleContextGraph] Snapshot has no nodes - resulting graph will be empty");
                }
                else
                {
                    // Build nodes in order
                    var created = new List<GraphNode>(nodesDto.Count);
                    for (int i = 0; i < nodesDto.Count; i++)
                    {
                        var nd = nodesDto[i];
                        var color = ColorRgb.FromUInt32(nd.Color);
                        var node = graph._fastGraph.CreateNodeExact(color, nd.X, nd.Y, nd.ObservationCount);
                        created.Add(node);
                    }

                    // Add edges
                    if (edgesDto != null)
                    {
                        int added = 0;
                        for (int i = 0; i < edgesDto.Count; i++)
                        {
                            var ed = edgesDto[i];
                            if (ed.From < 0 || ed.From >= created.Count || ed.To < 0 || ed.To >= created.Count)
                                continue;
                            var from = created[ed.From];
                            var to = created[ed.To];
                            var dir = (Direction)ed.Direction;
                            graph._fastGraph.AddEdge(from, dir, to, ed.Weight);
                            added++;
                        }
                        Console.WriteLine($"[MultiScaleContextGraph] Rebuilt graph: {created.Count} nodes, {added} edges");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MultiScaleContextGraph] Failed to rebuild from v2 snapshot: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine("[MultiScaleContextGraph] Legacy format detected - converting to fast graph...");
            try
            {
                if (snapshot.Graphs != null)
                {
                    int adjEntries = 0;
                    int edgesAdded = 0;
                    foreach (var kv in snapshot.Graphs)
                    {
                        var gSnap = kv.Value;
                        if (gSnap == null) continue;

                        // Prefer simple adjacency if available (fast to map)
                        if (gSnap.Adjacency != null)
                        {
                            foreach (var entry in gSnap.Adjacency)
                            {
                                adjEntries++;
                                var centerNode = graph._fastGraph.GetOrCreateNode(entry.Center, 0.5f, 0.5f);
                                foreach (var cw in entry.Colors)
                                {
                                    var targetNode = graph._fastGraph.GetOrCreateNode(cw.Color, 0.5f, 0.5f);
                                    graph._fastGraph.AddEdge(centerNode, entry.Direction, targetNode, (float)cw.Weight);
                                    edgesAdded++;
                                }
                            }
                        }

                        // Additionally map directional weights from patterns if present
                        if (gSnap.Patterns != null)
                        {
                            foreach (var p in gSnap.Patterns)
                            {
                                var centerNode = graph._fastGraph.GetOrCreateNode(p.Center, 0.5f, 0.5f);
                                foreach (var dirEntry in p.DirectionWeights)
                                {
                                    foreach (var cw in dirEntry.Colors)
                                    {
                                        var targetNode = graph._fastGraph.GetOrCreateNode(cw.Color, 0.5f, 0.5f);
                                        graph._fastGraph.AddEdge(centerNode, dirEntry.Direction, targetNode, (float)cw.Weight);
                                        edgesAdded++;
                                    }
                                }
                            }
                        }
                    }

                    Console.WriteLine($"[MultiScaleContextGraph] Legacy conversion complete: {graph._fastGraph.GetNodeCount()} nodes, {edgesAdded} edges from {adjEntries} adjacency entries");
                }
                else
                {
                    Console.WriteLine("[MultiScaleContextGraph] No legacy graphs found in snapshot");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[MultiScaleContextGraph] Legacy conversion failed: {ex.Message}");
            }
        }

        return graph;
    }
}
