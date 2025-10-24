# Neg2Posi

## Preview caching

Neg2Posi now keeps a small in-memory cache of preview results to make tone
adjustment sliders and crop edits much more responsive. Each cache entry stores
the original "before" image along with the heavy-weight processing output
before tone adjustments. Entries are keyed by the normalized source path, film
type, and current manual crop signature, and are automatically invalidated when
the source file timestamp or crop overrides change. The Qt UI flushes the cache
on exit to release memory.

## Packaging (macOS)

Build the standalone macOS bundle from a clean virtual environment so that
PyInstaller can collect the `liblensfun` dependencies shipped with
`lensfunpy`:

```bash
pyinstaller packaging/macos.spec --windowed --noconfirm
```

After building, test the bundle on a macOS system that does not have a
system-wide installation of lensfun to confirm that the GUI launches and that
lens correction features function as expected.
