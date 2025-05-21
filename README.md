# Tumor Allele Frquency Interpreter (TAFI)

Simulation-based algorithm that infers tumor purity, clonal structure, and evolutionary mode (Wright–Fisher vs. Exponential) from variant allele frequency (VAF) distributions in bulk sequencing data.

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

### Output

The output file stored in the results folder includes:

| Column Name   | Description |
|---------------|-------------|
| `donor`       | Donor identifier or name |
| `observed_n`  | Number of mutations in the input file |
| `cov`         | Average sequencing coverage |
| `min_reads`   | Minimum `AD_ALT` observed |
| `pur_WF`      | Tumor purity predicted by the WF model |
| `S_WF`        | Number of subclonal mutations predicted by the WF model |
| `C_WF`        | Number of clonal mutations predicted by the WF model |
| `score_WF`    | Score for the best WF model fit |
| `pur_EXP`     | Tumor purity predicted by the EXP model |
| `S_EXP`       | Number of subclonal mutations predicted by the EXP model |
| `C_EXP`       | Number of clonal mutations predicted by the EXP model |
| `score_EXP`   | Score for the best EXP model fit |

