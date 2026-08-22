using Aivqc.Trainer.Models;

namespace Aivqc.Trainer.Tests.Models;

public sealed class AnnotationColorPaletteTests
{
    [Fact]
    public void Create_AssignsNineDistinctColorsAndShortcuts()
    {
        var classes = Enumerable.Range(1, 9).Select(index => $"class-{index}").ToArray();

        var items = AnnotationColorPalette.Create(classes);

        Assert.Equal(Enumerable.Range(1, 9), items.Select(item => item.Shortcut));
        Assert.Equal(9, items.Select(item => item.ColorHex).Distinct().Count());
        Assert.Equal(classes, items.Select(item => item.ClassName));
    }

    [Fact]
    public void GetColor_IsStableForClassOrderAndCaseInsensitive()
    {
        string[] classes = ["scratch", "cut", "foreign-body"];

        var first = AnnotationColorPalette.GetColor("cut", classes);
        var sameClass = AnnotationColorPalette.GetColor("CUT", classes);
        var differentClass = AnnotationColorPalette.GetColor("scratch", classes);

        Assert.Equal(first, sameClass);
        Assert.NotEqual(first, differentClass);
    }
}
