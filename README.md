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
-r          recurse into input directories
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

For each PDF, `pdfcom` tries every non-empty permutation of the available
methods:

- Ghostscript as a subprocess, when `gs` is on `PATH`
- `pikepdf` as a library
- PyMuPDF as a library using the equivalent of `-ggggz`

Files that cannot be made smaller are copied unchanged in `-o` mode and left
alone in `-i` mode.

Use `-w` to process multiple input PDFs in parallel. The pool uses Python's
spawn multiprocessing context, so each worker process runs an independent PDF
compression pipeline.
