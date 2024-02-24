import argparse
import itertools
import numpy
import os
import pandas
from scipy import stats
from sklearn import neighbors
import sys

def multinomial_rvs(count, p, rng=None):
    """
    From: https://stackoverflow.com/questions/55818845/fast-vectorized-multinomial-in-python
    
    Sample from the multinomial distribution with multiple p vectors.

    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    out = numpy.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with numpy.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[numpy.isnan(condp) | (condp < 0)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = rng.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

def calc_biochemical_threshold(sim_func, locval, negative_selection=True):
    '''Calculate the "biochemical threshold" as the point of maximal downward curvature in the first
    bend of the inverse sigmoid function, calculated from the first and second derivatives of the
    inverse sigmoid with the formula y''/(1+y'^2)^(3/2).
    '''
    xvals = [elt for elt in numpy.arange(0,1,0.01) if elt < locval]
    if len(xvals) < 10:
        return locval
    yvals = numpy.array([sim_func(x) for x in xvals])
    mom1 = yvals[1:] - yvals[:-1]
    mom2 = mom1[1:] - mom1[:-1]
    curve = mom2/((1 + (mom1[1:]**2))**(3/2))
    curve_mom1 = curve[1:] - curve[:-1]
    if negative_selection is True:
        biochem_threshold = xvals[numpy.argmin(curve_mom1) + 3]
    else:
        biochem_threshold = xvals[numpy.argmax(curve_mom1) + 3]
    return biochem_threshold

def get_sim_func(loc, depth, pitch, inverse_sigmoid=True):
    '''Returns a lambda function respresenting either a sigmoid function (if inverse_sigmoid == False) or
    an inverse sigmoid (if inverse_sigmoid == True, the default) parameterized by the provided location,
    depth, and pitch parameters.
    '''
    if inverse_sigmoid is True:
        return lambda x:((1-depth)*(((10**pitch)**-(x-loc))/(1+(10**pitch)**-(x-loc)))) + depth
    else:
        return lambda x:((1-depth)*(((10**pitch)**(x-loc))/(1+(10**pitch)**(x-loc)))) + depth

def train_one_invsig_model(day0_cells, depth=0.57, loc=0.6, pitch=7, binomial_count=1000, start_seed=5,
                           passage_schedule={0:2500, 3:5000, 5:2500, 8:5000, 10:2500, 13:5000},
                           scilite_days=[0,5,10,15], het_cols=['LHON_het', 'SILENT_het', 'wt_het'],
                           tp_col='timepoint', doublings_per_day=1.0, negative_selection=True,
                           n_scilite_cells=1000):
    '''This function trains a single simulated culture, including producing simulated SCI-LITE data sets
    at the provided scilite_days.

    Parameters:
    day0_cells = Pandas DataFrame containing the observed SCI-LITE data for defining the heteroplasmy distribution of the initial time point of the culture.
    depth = The sigmoid depth parameter value
    loc = The sigmoid location parameter value
    pitch = The sigmoid pitch parameter value
    binomial_count = The mtDNA copy number to simulate. By default, each simulated cell will have this many UMIs drawn from a multinomial distribution parameterized by the heteroplasmy of each allele in that cell. If the 'umi_scale' column in the day0_cells DataFrame is present, then that will define for each cell a fraction of [binomial_count] UMIs to draw. When called by the 'search_space_prolif_invsig_kde_mse()' function, the umi_scale will be set so that the observed cell with the highest UMI coverage will get the full [binomial_count] number of draws, and every other cell will receive a fraction of that based on their observed UMI count relative to the highest coverage cell.
    start_seed = The seed for the random number generator.
    passage_schedule = A dict containing integer time points that should correspond to passages, along with the integer number of cells that should be preserved in each culture passage. The initial timepoint is required to be defined in this data structure, and this sets the initial number of cells in the simulation.
    scilite_days = A list of integers indicating at which days in the simulated culture to sample cells for a simulated SCI-LITE data point.
    het_cols = The columns in day0_cells that correspond to the heteroplasmy values of the different alleles of the locus being simulated. The values in these columns for each cell should sum to one.
    tp_col = The column name that should be used to store the integer time point for the simulated SCI-LITE data.
    doublings_per_day = A float indicating the number of times the culture should double in size during a simulated day.
    negative_selection = A boolean indicating whether the high heteroplasmic cells are under negative or positive selection. This parameter corrsponds to simulating the heteroplasmy threshold with either an inverse sigmoid function for negative selection or a regular sigmoid function for positive selection.
    n_scilite_cells = The number of simulated cells to draw for each SCI-LITE data point.
    '''
    #get random seeds
    rng = numpy.random.default_rng(seed=start_seed)
    rand_seeds = rng.integers(1, high=1000000000, size=2*scilite_days[-1])

    #get observed heteroplasmy values
    if het_cols is None:
        het_cols = day0_cells.columns[day0_cells.columns.str.endswith('_het')].to_list()
    if not numpy.all(day0_cells[het_cols].values.sum(axis=1) <= 1+1e-8):
        raise Exception(f'Heteroplasmy columns {het_cols} sum to more than one.')
    day0_cells['wt_het'] = 1 - day0_cells[het_cols].sum(axis=1)
    day0_cells.loc[day0_cells['wt_het'] > 1, 'wt_het'] = 0
    het_cols = het_cols + ['wt_het']

    #initialize list of synthetic SCI-LITE results
    synth_data_total = []

    #simulate day 0 based on the day 0 SCI-LITE data provided
    synth_data_tmp = day0_cells.sample(n=passage_schedule[scilite_days[0]], replace=True, random_state=rand_seeds[0])
    ncells = n_scilite_cells

    if 'umi_scale' in synth_data_tmp:
        num_draws = (synth_data_tmp['umi_scale'].to_numpy(copy=True) * binomial_count).round().astype(int)
    else:
        num_draws = (numpy.ones(synth_data_tmp.shape[0]) * binomial_count).astype(int)
    multnom_res = multinomial_rvs(num_draws,
                                  synth_data_tmp[het_cols].to_numpy(copy=True), rng=rng)
    synth_data_tmp[het_cols] = (multnom_res.T/multnom_res.sum(axis=1)).T

    #set the initial proliferation weights
    sim_func = get_sim_func(loc, depth, pitch, inverse_sigmoid=negative_selection)
    synth_data_tmp['proliferative_weight'] = [sim_func(x) for x in synth_data_tmp[het_cols[0]].to_numpy()]

    #save the initial synthetic data
    synth_data_tmp[tp_col] = scilite_days[0]
    synth_data_total.append(synth_data_tmp.sample(ncells, replace=True, random_state=rand_seeds[1]))

    #simulate the timecourse
    for idx in range(scilite_days[0], scilite_days[-1]):
        #simulate cells dividing in this 24 hr period
        synth_data_tmp = pandas.concat([synth_data_tmp,
                                        synth_data_tmp.sample(frac=doublings_per_day,
                                                              replace=True,
                                                              weights='proliferative_weight',
                                                              random_state=rand_seeds[1+idx])], ignore_index=True)
        #simulate small shifts in heteroplasmy over the cell division
        if 'umi_scale' in synth_data_tmp:
            num_draws = (synth_data_tmp['umi_scale'].to_numpy(copy=True) * binomial_count).round().astype(int)
        else:
            num_draws = (numpy.ones(synth_data_tmp.shape[0]) * binomial_count).astype(int)
        multnom_res = multinomial_rvs(num_draws,
                                      synth_data_tmp[het_cols].to_numpy(copy=True), rng=rng)
        synth_data_tmp[het_cols] = (multnom_res.T/multnom_res.sum(axis=1)).T

        #update the proliferation weights
        synth_data_tmp['proliferative_weight'] = [sim_func(x) for x in synth_data_tmp[het_cols[0]].to_numpy()]

        #take SCI-LITE sample if needed
        if idx+1 in scilite_days:
            synth_data_tmp[tp_col] = idx+1
            synth_data_total.append(synth_data_tmp.sample(ncells, replace=True, random_state=rand_seeds[idx]))

        #simulate passaging the cells based on the provided passaging schedule
        if idx+1 in passage_schedule:
            synth_data_tmp = synth_data_tmp.sample(n=passage_schedule[idx+1], replace=False,
                                                   random_state=rand_seeds[0-idx]).reset_index(drop=True)

    #combine the results into a single dataset
    synth_data_total = pandas.concat(synth_data_total)

    return (synth_data_total, calc_biochemical_threshold(sim_func, loc, negative_selection=negative_selection))


def search_space_prolif_invsig_kde_mse(all_obs, tp_col, depth_range=(0.05, 0.95),
                                       loc_range=(0.5, 0.8), pitch_range=(4,12),
                                       mtCN_opts=(500, 1000, 2000, 5000),
                                       iter_num=5000, init_seed=5,
                                       het_cols=None, #scilite_days=scilite_days,
                                       passage_schedule={0:2500, 3:5000, 5:2500, 8:5000, 10:2500, 13:5000},
                                       scaled_umi_sampling=True,
                                       doublings_per_day=1.0, negative_selection=True):
    ''' This function will conduct a random search over a range of values for the sigmoid function parameters to 
    find a set of parameters that fit the observed SCI-LITE data well.

    all_obs = Pandas DataFrame containing the full observed SCI-LITE data set. This will be used to calculate the mean squared error of the different models, and the earliest time point in this data structure will be used to initialize the models at each iteration.
    tp_col = The column of the all_obs DataFrame that contains time points as integers.
    depth_range = The range of values from which to draw proposed depth values during the random search. Restricted to [0,1.0].
    loc_range = The range of values from which to draw proposed location values during the random search. Restricted to [0,1.0]
    pitch_range = The range of values from which to draw proposed pitch values during the random search.
    mtCN_opts = A tuple of integers indicating the set of binomial_counts parameter values to try.
    iter_num = The number of random models to train and record performance for.
    init_seed = The seed for the random number generator that will be used to choose parameter values to test.
    het_cols = The all_obs columns that contain the heteroplasmy values for the alleles of interest in the simulation.
    passage_schedule = A dict containing integer time points that should correspond to passages, along with the integer number of cells that should be preserved in each culture passage. The initial timepoint is required to be defined in this data structure, and this sets the initial number of cells in the simulation.
    scaled_umi_sampling = A boolean indicating whether or not to create the 'umi_scale' column that will be used to scale the number of UMIs sampled for a given cell based on its relative UMI coverage [default: True]
    doublings_per_day = A float indicating the number of times the culture should double in size during a simulated day.
    negative_selection = A boolean indicating whether the high heteroplasmic cells are under negative or positive selection. This parameter corrsponds to simulating the heteroplasmy threshold with either an inverse sigmoid function for negative selection or a regular sigmoid function for positive selection.
    '''
    #get the observed time points and extract the initial time point
    obs_days = sorted(all_obs[tp_col].astype(int).unique())
    cols_to_keep = ['umi_count_for_filtering'] + het_cols
    initial_tp = all_obs.loc[all_obs[tp_col] == obs_days[0], cols_to_keep].copy()
    ncells = all_obs[tp_col].value_counts().values[-1]

    #Calculate UMI scaling factors for the initial timepoint, if requested
    if scaled_umi_sampling is True:
        initial_tp['umi_scale'] = initial_tp['umi_count_for_filtering']/initial_tp['umi_count_for_filtering'].max()

    #initialize random number generator and the data structure to track the optimization progress
    rng = numpy.random.default_rng(seed=init_seed)
    search_res = {'depth':[],
                  'loc':[],
                  'pitch':[],
                  'mtCN':[],
                  'seed':[],
                  'biochem_threshold':[],
                  'kde_mse':[],
                  'mean_het_mse':[]}
    for tp, allele in itertools.product(all_obs[tp_col].unique(), het_cols):
        search_res[f'day{tp}_{allele}_mean'] = []

    #calculate the observed KDE and the observed mean heteroplasmy values,
    # both of which will be used for calculating MSE values during the
    # optimization.
    x_d = numpy.meshgrid(*([numpy.linspace(0, 1.0, 100)]*len(het_cols)))
    x_d = numpy.array([x_d[idx].flatten() for idx in range(len(het_cols))]).T
    obs_kde_res = []
    for day_val in obs_days[1:]:
        # instantiate and fit the KDE model
        kde = neighbors.KernelDensity(bandwidth=0.05, kernel='gaussian')
        kde.fit(all_obs.loc[all_obs[tp_col] == day_val, het_cols].values)
        # score_samples returns the log of the probability density
        obs_kde_res.append(numpy.exp(kde.score_samples(x_d)))
    obs_kde_res = numpy.vstack(obs_kde_res)
    obs_means = numpy.hstack([all_obs.groupby(tp_col)[het_elt].mean()[obs_days].to_numpy()
                              for het_elt in het_cols])

    #Run the optimization
    for iter_idx in range(iter_num):
        if iter_idx and not iter_idx%10:
            print(f'Completed {iter_idx} iterations.')
        depthval = rng.uniform(*depth_range)
        locval = rng.uniform(*loc_range)
        pitchval = rng.uniform(*pitch_range)
        mtCNval = rng.choice(mtCN_opts)
        search_res['depth'].append(depthval)
        search_res['loc'].append(locval)
        search_res['pitch'].append(pitchval)
        search_res['mtCN'].append(mtCNval)
        search_res['seed'].append(iter_idx)

        #train the selected model
        iter_res, biochem_threshold = train_one_invsig_model(initial_tp, depth=depthval, loc=locval, pitch=pitchval,
                                                             binomial_count=mtCNval, start_seed=iter_idx,
                                                             scilite_days=obs_days,
                                                             passage_schedule=passage_schedule,
                                                             het_cols=het_cols, tp_col=tp_col, doublings_per_day=doublings_per_day, negative_selection=negative_selection, n_scilite_cells=ncells)
        search_res['biochem_threshold'].append(biochem_threshold)

        #calculate MSE values for this model; skip the inital_tp day when calculating the KDE MSE because
        # it is very similar to the observed data by construction.
        synth_kde_res = []
        for day_val in obs_days[1:]:
            # instantiate and fit the KDE model
            kde = neighbors.KernelDensity(bandwidth=0.05, kernel='gaussian')
            kde.fit(iter_res.loc[iter_res[tp_col] == day_val, het_cols].values)
            # score_samples returns the log of the probability density
            synth_kde_res.append(numpy.exp(kde.score_samples(x_d)))
        synth_kde_res = numpy.vstack(synth_kde_res)

        synth_means = numpy.hstack([iter_res.groupby(tp_col)[het_elt].mean()[obs_days].to_numpy()
                                    for het_elt in het_cols])
        for synth_means_idx, (tp, allele) in enumerate(itertools.product(obs_days, het_cols)):
            search_res[f'day{tp}_{allele}_mean'].append(synth_means[synth_means_idx])

        kde_mse_val = numpy.mean((obs_kde_res - synth_kde_res)**2)
        search_res['kde_mse'].append(kde_mse_val)
        mean_het_mse_val = numpy.mean((synth_means - obs_means)**2)
        search_res['mean_het_mse'].append(mean_het_mse_val)
    return pandas.DataFrame(search_res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--heteroplasmy-df-csv', help='The SCI-LITE heteroplasmy data to model. Assumed to be the result of running the SCI-LITE pipeline. The user must specify the columns that contain the heteroplasmy of the different alleles to model, and must also specify a user-generated column that contains the integer day on which the cells were collected, with zero indicating the initial time point and other integers representing the 1-based counts of days in culture.')
    parser.add_argument('--outdir', help='The output directory where the intermediate files should be saved, and ultimately where the final files should also be saved. Default is the same directory as the heteroplasmy_df_csv.')
    parser.add_argument('--iter-num', type=int, help='Number of iterations to run before checkpointing hyperparameter search to a file. [default: %(default)d]', default=500)
    parser.add_argument('--iter-total', type=int, help='Total number of models to train in this hyperparameter search. [default: %(default)d]', default=8000)
    parser.add_argument('--iter-count', type=int, help='The iteration number with which to start the hyperparameter search. Used in case the search is interrupted and needs to be restarted from the previous checkpoint. [default: %(default)d]', default=0)
    parser.add_argument('--loc-range', default='0.4,0.9', help='Comma-delimited pair of float values from the interval [0,1] that specifies the range from which to sample values for the inverse sigmoid location parameter. [default: %(default)s]')
    parser.add_argument('--depth-range', default='0.01,0.99', help='Comma-delimited pair of float values from the interval [0,1] that specifies the range from which to sample values for the inverse sigmoid depth parameter. [default: %(default)s]')
    parser.add_argument('--pitch-range', default='1.0,13.0', help='Comma-delimited pair of float values that specifies the range from which to sample values for the inverse sigmoid pitch parameter. Usually reasonable values for this parameter are within the interval (0,15]. [default: %(default)s]')
    parser.add_argument('--mtCN-opts', default='500,1000,2000,4000', help='Comma-delimited set of integers that represent the mtDNA copy number to simulate. Each model will use a randomly-chosen value from this set to test the sensitivity of the model fit to the number of simulated molecules per cell. The results are usually not very sensitive to this setting. [default: %(default)s]')
    parser.add_argument('--heteroplasmy-columns', help='Comma-delimited list of the columns containing heteroplasmy values for the alternative alleles to be simulated. The first specified heteroplasmy value should be for the pathogenic allele (i.e. the one determining the shape of the inverse sigmoid). Any other unspecified alleles, for example WT alleles, will be lumped together and calculated as one minus the sum of the provided alternate allele heteroplasmies.')
    parser.add_argument('--timepoint-column', help='The name of the column containing the integer days on which the SCI-LITE data were collected. These days will be used to compare the simulation results to the observed data and calculate the mean squared error values.')
    parser.add_argument('--passage-schedule', help='Comma-delimited list of days on which to simulate passaging the culture and the number of cells to sample when passaging on that day. For example, to passage on an alternating schedule of two days and three days, with passaging 2500 cells after two days of growth and 5000 cells after three days of growth, one could specify this string: 0:2500,3:5000,5:2500,8:5000,10:2500,13:5000. The initial timepoint is the only required one. It should be designated as zero, and the number of cells specified will initialize the simulated culture.')
    parser.add_argument('--doublings-per-day', type=float, default=1.0, help='The number of times per day that the culture doubles in size. Fractional values are ok. [default: %(default)d]')
    parser.add_argument('--no-umi-per-cell-scaling', default=False, action='store_true', help='By default, the modeling scales the number of simulated UMIs based on the distribution of UMI counts per cell in the observed data. This avoids giving undue confidence to heteroplasmy estimates from cells with low numbers of UMIs. Specify this option to skip the scaling and simply sample the same number of UMIs for every simulated cell.')
    parser.add_argument('--positive-selection', default=False, action='store_true', help='If this option is specified, use a regular sigmoid to implement a selective advantage for cells with high heteroplasmy. By default the model uses an inverse sigmoid under the assumption that the cells with high heteroplasmy have a fitness defect relative to the low heteroplasmy cells.')
    args = parser.parse_args()

    #get single cell heteroplasmy estimates
    scilite_estimates = pandas.read_csv(args.heteroplasmy_df_csv)
    scilite_estimates.columns = scilite_estimates.columns.str.replace('_HET', '_het')

    #get output directory
    if args.outdir is None:
        args.outdir = os.path.dirname(args.heteroplasmy_df_csv)

    #get optimization parameters
    loc_range = [float(elt) for elt in args.loc_range.split(',') if (float(elt) >= 0) and (float(elt) <= 1)]
    if len(loc_range) != 2:
        sys.exit(f'--loc-range argument must specify two floats on the interval [0,1]. Provided value {args.loc_range} is invalid.')
    depth_range = [float(elt) for elt in args.depth_range.split(',') if (float(elt) >= 0) and (float(elt) <= 1)]
    if len(depth_range) != 2:
        sys.exit(f'--depth-range argument must specify two floats on the interval [0,1]. Provided value {args.depth_range} is invalid.')
    pitch_range = [float(elt) for elt in args.pitch_range.split(',')]
    if len(loc_range) != 2:
        sys.exit(f'--pitch-range argument must specify two floats. Provided value {args.pitch_range} is invalid.')
    mtCN_opts = [int(elt) for elt in args.mtCN_opts.split(',')]
    het_cols = None if args.heteroplasmy_columns is None else args.heteroplasmy_columns.split(',')
    tp_col = args.timepoint_column

    if args.passage_schedule is None:
        passage_schedule = {}
    else:
        passage_schedule = dict([tuple([int(inner_elt) for inner_elt in elt.split(':')])
                                 for elt in args.passage_schedule.split(',')])

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    iter_num = args.iter_num
    iter_count = args.iter_count
    iter_res_paths = []
    while iter_count < args.iter_total:
        test = search_space_prolif_invsig_kde_mse(scilite_estimates, iter_num=iter_num, init_seed=iter_count,
                                                  depth_range=depth_range, loc_range=loc_range,
                                                  pitch_range=pitch_range, mtCN_opts=mtCN_opts,
                                                  tp_col=tp_col, het_cols=het_cols,
                                                  passage_schedule=passage_schedule,
                                                  scaled_umi_sampling=not args.no_umi_per_cell_scaling,
                                                  doublings_per_day=args.doublings_per_day,
                                                  negative_selection=not args.positive_selection)

        out_path = os.path.join(args.outdir, f'fit_inverse_sigmoid.{iter_count}.csv')
        test.to_csv(out_path, index=False, sep=',')
        iter_res_paths.append(out_path)
        print(out_path)
        iter_count += iter_num

    #merge output files and calculate the combined MSE
    total_res = pandas.concat([pandas.read_csv(elt) for elt in iter_res_paths])
    total_res['combined_mse'] = (((total_res['kde_mse'] - total_res['kde_mse'].min())
                                  /(total_res['kde_mse'].max() - total_res['kde_mse'].min()))
                                 + 2*((total_res['mean_het_mse'] - total_res['mean_het_mse'].min())
                                      /(total_res['mean_het_mse'].max() - total_res['mean_het_mse'].min())))
    #now smooth the combined MSE with by taking the average of the 50 nearest neighbors
    for_nn = total_res[['depth', 'loc', 'pitch']].values
    for_nn -= for_nn.min(axis=0)
    for_nn /= for_nn.max(axis=0)
    nn_obj = neighbors.NearestNeighbors(n_neighbors=50).fit(for_nn)
    total_res['smoothed_mse'] = [total_res.iloc[nn_obj.kneighbors(elt.reshape(1,-1))[1].flatten()]['combined_mse'].mean() for elt in for_nn]

    #save the total results with the new combined MSE and smoothed MSE columns
    out_path = os.path.join(args.outdir, 'fit_inverse_sigmoid.all_iterations.csv')
    total_res.to_csv(out_path, index=False, sep=',')
    for path in iter_res_paths:
        os.remove(path)

    #print out the best model parameters
    best_model = total_res.iloc[total_res['smoothed_mse'].argmin()]
    print('Best model:')
    print(best_model)
    best_model.to_csv(os.path.join(args.outdir, 'fit_inverse_sigmoid.best_model.csv'), header=False)
