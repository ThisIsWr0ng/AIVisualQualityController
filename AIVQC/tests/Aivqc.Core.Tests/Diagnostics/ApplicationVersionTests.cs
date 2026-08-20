using System.Reflection;
using Aivqc.Core.Diagnostics;

namespace Aivqc.Core.Tests.Diagnostics;

public sealed class ApplicationVersionTests
{
    [Fact]
    public void FromAssembly_ReturnsInformationalVersion()
    {
        var expected = typeof(ApplicationVersionTests).Assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()!
            .InformationalVersion;

        var actual = ApplicationVersion.FromAssembly(typeof(ApplicationVersionTests).Assembly);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void DisplayFromAssembly_RemovesBuildMetadata()
    {
        var actual = ApplicationVersion.DisplayFromAssembly(typeof(ApplicationVersionTests).Assembly);

        Assert.DoesNotContain('+', actual);
    }
}
