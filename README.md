# scilite-analysis
A repository to host the pipeline parameters and data analysis code from Kotrys, et al. Single-cell analysis reveals context-dependent, cell-level selection of mtDNA. _Nature_. 2024.

## Dependencies
Most notebooks in this repository were run with the following package versions:
* Python v3.7.12
* matplotlib 3.4.2
* numpy 1.21.0
* pandas 1.1.5
* plotly 5.16.1
* pysam 0.16.0.1
* scikit-learn 0.23.1
* scipy 1.7.0
* seaborn 0.11.1

The Kimura analysis in SCI-LITE_LHON_SILENT_timecourse/kimura_analysis.ipynb used the following:
* Python 3.10.12
* matplotlib 3.8.0
* numpy 1.24.4
* pandas 2.1.1
* rpy2 3.5.11
* seaborn 0.13.0
* R 4.2.3
* heteroplasmy 0.0.2.1
* kimura 0.0.0.9001

To handle conflicting dependencies, we recommend using a virtual environments, such as provided in package managers like [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

## How to use this repository.
Each subdirectory in this repository contains the parameters and auxiliary files to run the SCI-LITE pipeline and reproduce the 
figures from Kotrys, et al. _Nature_. 2024. Please use the following steps:

1) Check out this repository and ensure the dependencies are installed.
1) Check out the [SCI-LITE pipeline repository](https://github.com/MoothaLab/scilite-pipeline) and install its dependencies.
1) Create a "./fastq" subdirectory under each of this repository's directories. Download the raw data in FASTQ format from the Sequence Read Archive (BioProject accession XXXXX) to the corresponding newly-created "fastq" directories. 
1) Once the raw FASTQ files are in place, then the SCI-LITE pipeline can be run using the command lines provided by the repository in "cmd_line.txt" files. Note that for the SRA files, the well_ID attributes are appended to the end of the FASTQ read name field with a space. The SCI-LITE pipeline will automatically parse this information from the read name instead of the file name if you run the code with the "--isWellIDFromRead" option.
1) After each pipeline run completes, the analysis code in the Jupyter notebooks can be run to reproduce the figures from the paper.

## Reproducing the LHON/SILENT heteroplasmy modeling.
To reproduce the inverse sigmoid heteroplasmy modeling directly, please download the raw data and run the SCI-LITE pipeline as described above, and then fit the inverse sigmoid model using the SCI-LITE_LHON_SILENT_timecourse/fit_inverse_sigmoid.py script. Note that this version of the script has certain values hard-coded that are specific to the LHON/SILENT time course analysis. If you would like to fit an inverse sigmoid heteroplasmy model to arbitrary SCI-LITE data, please use the more general fit_sigmoid.py script in the root of this repository.

## Usage text for fit_sigmoid_usage.py:
```
usage: fit_sigmoid.py [-h] [--heteroplasmy-df-csv HETEROPLASMY_DF_CSV]
                      [--outdir OUTDIR] [--iter-num ITER_NUM]
                      [--iter-total ITER_TOTAL] [--iter-count ITER_COUNT]
                      [--loc-range LOC_RANGE] [--depth-range DEPTH_RANGE]
                      [--pitch-range PITCH_RANGE] [--mtCN-opts MTCN_OPTS]
                      [--heteroplasmy-columns HETEROPLASMY_COLUMNS]
                      [--timepoint-column TIMEPOINT_COLUMN]
                      [--passage-schedule PASSAGE_SCHEDULE]
                      [--doublings-per-day DOUBLINGS_PER_DAY]
                      [--no-umi-per-cell-scaling] [--positive-selection]

optional arguments:
  -h, --help            show this help message and exit
  --heteroplasmy-df-csv HETEROPLASMY_DF_CSV
                        The SCI-LITE heteroplasmy data to model. Assumed to be
                        the result of running the SCI-LITE pipeline. The user
                        must specify the columns that contain the heteroplasmy
                        of the different alleles to model, and must also
                        specify a user-generated column that contains the
                        integer day on which the cells were collected, with
                        zero indicating the initial time point and other
                        integers representing the 1-based counts of days in
                        culture.
  --outdir OUTDIR       The output directory where the intermediate files
                        should be saved, and ultimately where the final files
                        should also be saved. Default is the same directory as
                        the heteroplasmy_df_csv.
  --iter-num ITER_NUM   Number of iterations to run before checkpointing
                        hyperparameter search to a file. [default: 500]
  --iter-total ITER_TOTAL
                        Total number of models to train in this hyperparameter
                        search. [default: 8000]
  --iter-count ITER_COUNT
                        The iteration number with which to start the
                        hyperparameter search. Used in case the search is
                        interrupted and needs to be restarted from the
                        previous checkpoint. [default: 0]
  --loc-range LOC_RANGE
                        Comma-delimited pair of float values from the interval
                        [0,1] that specifies the range from which to sample
                        values for the inverse sigmoid location parameter.
                        [default: 0.4,0.9]
  --depth-range DEPTH_RANGE
                        Comma-delimited pair of float values from the interval
                        [0,1] that specifies the range from which to sample
                        values for the inverse sigmoid depth parameter.
                        [default: 0.01,0.99]
  --pitch-range PITCH_RANGE
                        Comma-delimited pair of float values that specifies
                        the range from which to sample values for the inverse
                        sigmoid pitch parameter. Usually reasonable values for
                        this parameter are within the interval (0,15].
                        [default: 1.0,13.0]
  --mtCN-opts MTCN_OPTS
                        Comma-delimited set of integers that represent the
                        mtDNA copy number to simulate. Each model will use a
                        randomly-chosen value from this set to test the
                        sensitivity of the model fit to the number of
                        simulated molecules per cell. The results are usually
                        not very sensitive to this setting. [default:
                        500,1000,2000,4000]
  --heteroplasmy-columns HETEROPLASMY_COLUMNS
                        Comma-delimited list of the columns containing
                        heteroplasmy values for the alternative alleles to be
                        simulated. The first specified heteroplasmy value
                        should be for the pathogenic allele (i.e. the one
                        determining the shape of the inverse sigmoid). Any
                        other unspecified alleles, for example WT alleles,
                        will be lumped together and calculated as one minus
                        the sum of the provided alternate allele
                        heteroplasmies.
  --timepoint-column TIMEPOINT_COLUMN
                        The name of the column containing the integer days on
                        which the SCI-LITE data were collected. These days
                        will be used to compare the simulation results to the
                        observed data and calculate the mean squared error
                        values.
  --passage-schedule PASSAGE_SCHEDULE
                        Comma-delimited list of days on which to simulate
                        passaging the culture and the number of cells to
                        sample when passaging on that day. For example, to
                        passage on an alternating schedule of two days and
                        three days, with passaging 2500 cells after two days
                        of growth and 5000 cells after three days of growth,
                        one could specify this string:
                        0:2500,3:5000,5:2500,8:5000,10:2500,13:5000. The
                        initial timepoint is the only required one. It should
                        be designated as zero, and the number of cells
                        specified will initialize the simulated culture.
  --doublings-per-day DOUBLINGS_PER_DAY
                        The number of times per day that the culture doubles
                        in size. Fractional values are ok. [default: 1]
  --no-umi-per-cell-scaling
                        By default, the modeling scales the number of
                        simulated UMIs based on the distribution of UMI counts
                        per cell in the observed data. This avoids giving
                        undue confidence to heteroplasmy estimates from cells
                        with low numbers of UMIs. Specify this option to skip
                        the scaling and simply sample the same number of UMIs
                        for every simulated cell.
  --positive-selection  If this option is specified, use a regular sigmoid to
                        implement a selective advantage for cells with high
                        heteroplasmy. By default the model uses an inverse
                        sigmoid under the assumption that the cells with high
                        heteroplasmy have a fitness defect relative to the low
                        heteroplasmy cells.
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). You are free to share or adapt the material for non-commercial purposes.

If you find this work helpful, please cite:
```
@article{kotrys2023nature,
  title = {Single-cell analysis reveals context-dependent, cell-level selection of mtDNA},
  author = {Kotrys, Anna V. and
            Durham, Timothy J. and
            Guo, Xiaoyan A. and
            Vantaku, Venkata R. and
            Parangi, Sareh  and
            Vamsi, Mootha K.},
  journal = {Nature},
  elocation-id = {...},
  doi = {...},
  publisher = {Springer Nature},
  URL = {...},
  eprint = {...},
  year = {2024},
}
```
