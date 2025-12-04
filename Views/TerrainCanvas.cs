using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using System;

namespace NNImage.Views;

/// <summary>
/// Interactive canvas for terrain editing with real-time brush preview
/// </summary>
public class TerrainCanvas : Control
{
    private WriteableBitmap? _terrainBitmap;
    private Point? _lastMousePos;
    private bool _isDrawing;

    public event Action<int, int, bool>? BrushStroke;
    public event Action<int, int>? MouseMove;

    public TerrainCanvas()
    {
        ClipToBounds = true;
        Focusable = true;
    }

    public void UpdateTerrain(uint[] pixels, int width, int height)
    {
        _terrainBitmap?.Dispose();
        _terrainBitmap = CreateBitmapFromPixels(pixels, width, height);
        InvalidateVisual();
    }

    private WriteableBitmap CreateBitmapFromPixels(uint[] pixels, int width, int height)
    {
        var bitmap = new WriteableBitmap(
            new PixelSize(width, height),
            new Vector(96, 96),
            Avalonia.Platform.PixelFormat.Bgra8888,
            Avalonia.Platform.AlphaFormat.Premul
        );

        using (var buffer = bitmap.Lock())
        {
            unsafe
            {
                var ptr = (uint*)buffer.Address.ToPointer();
                for (int i = 0; i < pixels.Length; i++)
                {
                    ptr[i] = pixels[i];
                }
            }
        }

        return bitmap;
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        var point = e.GetPosition(this);
        _isDrawing = true;
        _lastMousePos = point;

        BrushStroke?.Invoke((int)point.X, (int)point.Y, true);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        var point = e.GetPosition(this);

        MouseMove?.Invoke((int)point.X, (int)point.Y);

        if (_isDrawing)
        {
            BrushStroke?.Invoke((int)point.X, (int)point.Y, false);
            _lastMousePos = point;
            InvalidateVisual();
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _isDrawing = false;
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);

        if (_terrainBitmap != null)
        {
            var destRect = new Rect(0, 0, Bounds.Width, Bounds.Height);
            context.DrawImage(_terrainBitmap, destRect);
        }

        // Draw brush cursor
        if (_lastMousePos.HasValue)
        {
            var center = _lastMousePos.Value;
            var brushRadius = 20; // Get from brush size

            var pen = new Pen(Brushes.White, 2);
            context.DrawEllipse(null, pen, center, brushRadius, brushRadius);
        }
    }
}
