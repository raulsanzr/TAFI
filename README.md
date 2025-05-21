# Tumor Allele Frquency Interpreter (TAFI)

Mutations that accumulate during tumor progression are footprints of its evolutionary history. The distribution of variant allele frequencies (**VAF**) can be used to infer relevant biological parameters or distinguish between tumor growth models.


## Installation

```bash
git clone https://github.com/raulsanzr/TAFI.git
```

## Usage

```bash
cd TAFI/scripts
python3 tafi.py input.csv
```

### Input

The input file must be a comma-separated values (CSV) file with a header containing the following two columns:

| Column Name | Description |
|-------------|-------------|
| `AD_REF`    | Allelic depth for the reference allele |
| `AD_ALT`    | Allelic depth for the alternative allele |

