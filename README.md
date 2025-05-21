# Tumor Allele Frquency Interpreter (TAFI)

Mutations that accumulate during tumor progression are footprints of its evolutionary history. The distribution of variant allele frequencies (**VAF**) can be used to infer relevant biological parameters or distinguish between tumor growth models.


## Installation

```bash
git clone https://github.com/raulsanzr/TAFI.git
```

## Usage

```bash
cd TAFI/scripts
python3 tafi.py path/input.csv
```

### Input

The input file must be a comma-separated values (CSV) file with a header containing the following two columns:

| Column Name | Description |
|-------------|-------------|
| `AD_REF`    | Allelic depth for the reference allele |
| `AD_ALT`    | Allelic depth for the alternative allele |

## Output

The output file stored in the results folder includes:

| Column Name   | Description |
|---------------|-------------|
| `donor`       | Donor identifier or name |
| `observed_n`  | Number of mutations in the input file |
| `cov`         | Average sequencing coverage |
| `min_reads`   | Minimum allelic depth of the alternative allele (`AD_ALT`) |
| `pur_WF`      | Tumor purity predicted by the WF model |
| `S_WF`        | Number of subclonal mutations (S) predicted by the WF model |
| `C_WF`        | Number of clonal mutations (C) predicted by the WF model |
| `score_WF`    | Score (distance) for the best WF model fit |
| `pur_EXP`     | Tumor purity predicted by the EXP model |
| `S_EXP`       | Number of subclonal mutations (S) predicted by the EXP model |
| `C_EXP`       | Number of clonal mutations (C) predicted by the EXP model |
| `score_EXP`   | Score (distance) for the best EXP model fit |

