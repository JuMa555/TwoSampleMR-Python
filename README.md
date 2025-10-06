# TwoSampleMR-Python
A package for performing Mendelian randomization using GWAS summary data. It allows causal inference using genetic variants as instrumental variables, fully integrated with the [IEU OpenGWAS](https://opengwas.io/) database.


This repository contains a reimplementation of the [TwoSampleMR](https://github.com/MRCIEU/TwoSampleMR) package.  
It was developed as part of a master's thesis at UniZg-FER for academic purposes and learning.  




| Module | Description |
|--------|--------------|
| `instruments.py` | Extracts instruments from OpenGWAS. |
| `query.py` | Queries outcome data from OpenGWAS. |
| `harmonise.py` | Harmonises exposure and outcome datasets. |
| `mr.py` | Core MR methods (IVW, Egger, Median, Mode, GRIP). |
| `heterogeneity.py` | Heterogeneity and pleiotropy tests. |
| `multivariable_mr.py` | Multivariable MR framework. |

## Example Usage:

```python
from instruments import extract_instruments
from query import extract_outcome_data
from harmonise import harmonise_data
from mr import mr

exposure = extract_instruments(["ieu-a-2"])
outcome = extract_outcome_data(exposure["SNP"], ["ieu-a-7"])
dat = harmonise_data(exposure, outcome)
res = mr(dat)
print(res)
```
