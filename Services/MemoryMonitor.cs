using System;
using System.Diagnostics;
using System.Threading;

namespace NNImage.Services;

/// <summary>
/// Monitors system memory usage and triggers streaming/pausing when approaching RAM limit
/// Prevents system-wide freezes by managing memory proactively
/// </summary>
public class MemoryMonitor
{
    private readonly long _maxRamBytes;
    private readonly long _streamingThresholdBytes;
    private readonly long _pauseThresholdBytes;
    private long _lastMemoryCheck = 0;
    private readonly long _checkIntervalTicks;

    // Performance counters for accurate system-wide RAM measurement
    private readonly PerformanceCounter? _ramCounter;
    private bool _useProcessMemory = false;

    public MemoryMonitor(int maxRamGB = 30)
    {
        _maxRamBytes = (long)maxRamGB * 1024 * 1024 * 1024;
        _streamingThresholdBytes = (long)(_maxRamBytes * 0.85); // Start streaming at 85%
        _pauseThresholdBytes = (long)(_maxRamBytes * 0.95); // Pause at 95%
        _checkIntervalTicks = Stopwatch.Frequency / 10; // Check 10 times per second max

        Console.WriteLine($"[MemoryMonitor] Initialized with {maxRamGB}GB limit");
        Console.WriteLine($"[MemoryMonitor] Streaming threshold: {_streamingThresholdBytes / (1024 * 1024 * 1024)}GB ({(_streamingThresholdBytes * 100.0 / _maxRamBytes):F1}%)");
        Console.WriteLine($"[MemoryMonitor] Pause threshold: {_pauseThresholdBytes / (1024 * 1024 * 1024)}GB ({(_pauseThresholdBytes * 100.0 / _maxRamBytes):F1}%)");

        // Try to create performance counter for system-wide RAM monitoring
        try
        {
            _ramCounter = new PerformanceCounter("Memory", "Available MBytes");
            _ramCounter.NextValue(); // Initialize
            Console.WriteLine("[MemoryMonitor] Using system-wide RAM monitoring");
        }
        catch
        {
            _useProcessMemory = true;
            Console.WriteLine("[MemoryMonitor] Falling back to process memory monitoring");
        }
    }

    /// <summary>
    /// Check current memory usage. Returns true if we should continue, false if we need to pause.
    /// </summary>
    public MemoryStatus CheckMemory()
    {
        var now = Stopwatch.GetTimestamp();

        // Rate limit checks for performance
        if (now - _lastMemoryCheck < _checkIntervalTicks)
        {
            return MemoryStatus.Normal;
        }

        _lastMemoryCheck = now;

        long usedBytes;

        if (_useProcessMemory)
        {
            // Fallback: use process memory
            using var process = Process.GetCurrentProcess();
            usedBytes = process.WorkingSet64;
        }
        else if (_ramCounter != null)
        {
            // Primary: use system-wide available RAM
            var availableMB = _ramCounter.NextValue();
            var totalRAM = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            usedBytes = totalRAM - ((long)availableMB * 1024 * 1024);
        }
        else
        {
            // Last resort: GC memory info
            usedBytes = GC.GetTotalMemory(false);
        }

        var usedPercent = (usedBytes * 100.0) / _maxRamBytes;

        if (usedBytes >= _pauseThresholdBytes)
        {
            Console.WriteLine($"[MemoryMonitor] ⚠ CRITICAL RAM: {usedBytes / (1024 * 1024 * 1024):F2}GB ({usedPercent:F1}%) - PAUSING");
            return MemoryStatus.Critical;
        }
        else if (usedBytes >= _streamingThresholdBytes)
        {
            Console.WriteLine($"[MemoryMonitor] ⚠ High RAM: {usedBytes / (1024 * 1024 * 1024):F2}GB ({usedPercent:F1}%) - STREAMING");
            return MemoryStatus.Streaming;
        }

        return MemoryStatus.Normal;
    }

    /// <summary>
    /// Wait for memory to drop below streaming threshold
    /// </summary>
    public void WaitForMemoryRecovery(Action<string>? statusCallback = null)
    {
        Console.WriteLine("[MemoryMonitor] Waiting for memory recovery...");
        statusCallback?.Invoke("⚠ High memory - waiting for GC...");

        var startTime = Stopwatch.GetTimestamp();
        var gcAttempts = 0;

        while (true)
        {
            // Force aggressive GC
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            gcAttempts++;

            Thread.Sleep(500); // Give system time to release memory

            var status = CheckMemory();
            if (status == MemoryStatus.Normal)
            {
                var elapsedSec = (Stopwatch.GetTimestamp() - startTime) / (double)Stopwatch.Frequency;
                Console.WriteLine($"[MemoryMonitor] Memory recovered after {elapsedSec:F1}s ({gcAttempts} GC cycles)");
                statusCallback?.Invoke($"✓ Memory recovered ({elapsedSec:F1}s)");
                return;
            }

            if (status == MemoryStatus.Streaming)
            {
                Console.WriteLine($"[MemoryMonitor] Memory at streaming level - continuing with caution");
                statusCallback?.Invoke("⚠ Memory at streaming level");
                return;
            }

            // Continue waiting if still critical
            var elapsed = (Stopwatch.GetTimestamp() - startTime) / (double)Stopwatch.Frequency;
            if (elapsed > 30) // Max wait time
            {
                Console.WriteLine($"[MemoryMonitor] ⚠ Memory recovery timeout after {elapsed:F1}s");
                statusCallback?.Invoke("⚠ Memory recovery timeout - continuing with risk");
                return;
            }

            statusCallback?.Invoke($"⚠ Waiting for memory... ({elapsed:F0}s)");
        }
    }

    public void Dispose()
    {
        _ramCounter?.Dispose();
    }
}

public enum MemoryStatus
{
    Normal,      // < 85% - normal operation
    Streaming,   // 85-95% - enable streaming, reduce batch sizes
    Critical     // > 95% - pause and wait for GC
}
