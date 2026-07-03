# pdfcom

`pdfcom` tries multiple PDF compression pipelines for each input PDF and keeps
the smallest valid result. It validates each candidate by reopening the PDF and
checking that the page count did not change.

```bash
pdfcom [options] <pdf_file_or_dir> [pdf_file_or_dir...]
```

## Options

```text
-n          dry run
-p <spec>   use a compression plan spec
-r          recurse into input directories
-s <file>   write per-plan optimization stats as JSON
-w <n>      use n worker processes
-i          replace reduced PDFs in place
-o <dir>    write outputs to <dir>
-c          clobber existing output files
-C          do not clobber existing output files; overrides -c
```

`-i` and `-o` are mutually exclusive. One of them is required unless `-n` is
used for analysis only.

When `-o` is used with directory inputs, output PDFs are prefixed with the input
directory name. With `-r`, the prefix can include several directories:

```bash
pdfcom -r -o compressed "Company Documents"
# Company Documents/Medical Plans/foo.pdf
# -> compressed/Company Documents/Medical Plans/foo.pdf
```

File inputs are written directly under the output directory:

```bash
pdfcom -o compressed foo.pdf
# -> compressed/foo.pdf
```

## Compression Methods

For each PDF, `pdfcom` tries the selected compression plans and keeps the
smallest valid result.

- Ghostscript as a subprocess, when `gs` is on `PATH`
- `pikepdf` as a library
- PyMuPDF as a library using the equivalent of `-ggggz`
- `qpdf` as a subprocess, when `qpdf` is on `PATH`

Files that cannot be made smaller are copied unchanged in `-o` mode and left
alone in `-i` mode.

Use `-w` to process multiple input PDFs in parallel. The pool uses Python's
spawn multiprocessing context, so each worker process runs an independent PDF
compression pipeline. Results are printed as workers complete.

The default plan spec is:

```text
gs?,{mu?,pike?},q
```

That tries Ghostscript in the first position or omits it, tries all
permutations of PyMuPDF and pikepdf in the middle including either one or
neither, and applies qpdf last. If `gs` or `qpdf` is not available on `PATH`,
that method is omitted from the generated plans. The qpdf stage recompresses
Flate streams with compression level 9 and generates object streams, but does
not enable lossy image optimization.

Plan specs use these method names:

```text
gs    Ghostscript
mu    PyMuPDF
pike  pikepdf
q     qpdf
```

Use commas for fixed order, braces for permutations, parentheses for fixed-order
combinations, and `?` to make the preceding item or group optional:

```text
gs,q              # gs then q
gs,q?             # gs+q and gs
{mu,pike}         # mu+pike and pike+mu
(mu,pike)         # mu+pike only
{mu?,pike?}       # both orders, either single method, or neither
gs?,{mu?,pike?},q # default
```

Use `-s` to write a JSON stats file with one result per input PDF. Each result
includes the winning plan, unavailable methods, failed plans, and per-plan
sizes so method order and marginal savings can be analyzed after a run.
