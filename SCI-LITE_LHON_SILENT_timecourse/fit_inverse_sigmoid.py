import argparse
import numpy
import os
import pandas
from sklearn import neighbors

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


def train_one_invsig_model(day0_cells, depth=0.57, loc=0.6, pitch=7, binomial_count=1000, start_seed=5):
    #get random seeds
    rng = numpy.random.default_rng(seed=start_seed)
    rand_seeds = rng.integers(1, high=1000000000, size=25)

    #generate cells
    synth_data = day0_cells.copy()
    ncells = day0_cells.shape[0]

    #simulate day 0 with a larger set of cells
    synth_data_tmp = day0_cells.sample(n=2500, replace=True, random_state=rand_seeds[0])

    het_cols = ['LHON_het', 'SILENT_ONLY_het', 'wt_het']
    if 'mtDNA_umi_scale' in synth_data_tmp:
        num_draws = (synth_data_tmp['mtDNA_umi_scale'].to_numpy(copy=True) * binomial_count).round().astype(int)
    else:
        num_draws = (numpy.ones(synth_data_tmp.shape[0]) * binomial_count).astype(int)
    multnom_res = multinomial_rvs(num_draws,
                                  synth_data_tmp[het_cols].to_numpy(copy=True), rng=rng)
    synth_data_tmp[het_cols] = (multnom_res.T/multnom_res.sum(axis=1)).T

    #set the initial proliferation weights
    sim_func = lambda x:((1-depth)*(((10**pitch)**-(x-loc))/(1+(10**pitch)**-(x-loc)))) + depth
    synth_data_tmp['proliferative_weight'] = [sim_func(x) for x in synth_data_tmp['LHON_het'].to_numpy()]
    synth_data_0d = synth_data_tmp.sample(ncells, replace=True, random_state=rand_seeds[1])
    synth_data_5d = None
    synth_data_10d = None
    for idx in range(15):
        #simulate passaging the cells every 3 or 2 days alternating (Anna's passaging schedule)
        if idx+1 in [5, 10]:
            synth_data_tmp = synth_data_tmp.sample(n=2500, replace=False,
                                                   random_state=rand_seeds[0-idx]).reset_index(drop=True)
        elif idx+1 in [3, 8, 13]:
            synth_data_tmp = synth_data_tmp.sample(n=5000, replace=False,
                                                   random_state=rand_seeds[0-idx]).reset_index(drop=True)

        #simulate cells dividing in this 24 hr period
        synth_data_tmp = pandas.concat([synth_data_tmp,
                                        synth_data_tmp.sample(frac=1.0, replace=True,
                                                              weights='proliferative_weight',
                                                              random_state=rand_seeds[3+idx])], ignore_index=True)
        #simulate small shifts in heteroplasmy over the cell division
        if 'mtDNA_umi_scale' in synth_data_tmp:
            num_draws = (synth_data_tmp['mtDNA_umi_scale'].to_numpy(copy=True) * binomial_count).round().astype(int)
        else:
            num_draws = (numpy.ones(synth_data_tmp.shape[0]) * binomial_count).astype(int)
        het_cols = ['LHON_het', 'SILENT_ONLY_het', 'wt_het']
        multnom_res = multinomial_rvs(num_draws,
                                      synth_data_tmp[het_cols].to_numpy(copy=True), rng=rng)
        synth_data_tmp[het_cols] = (multnom_res.T/multnom_res.sum(axis=1)).T
        #update the proliferation weights
        synth_data_tmp['proliferative_weight'] = [sim_func(x) for x in synth_data_tmp['LHON_het'].to_numpy()]
        if idx == 4:
            synth_data_5d = synth_data_tmp.sample(ncells, replace=True, random_state=rand_seeds[20])
        elif idx == 9:
            synth_data_10d = synth_data_tmp.sample(ncells, replace=True, random_state=rand_seeds[21])
    else:
        synth_data_15d = synth_data_tmp.sample(ncells, replace=True, random_state=rand_seeds[22])

    #combine the results into a single dataset
    synth_data_0d['timepoint'] = 'day 0'
    synth_data_5d['timepoint'] = 'day 5'
    synth_data_10d['timepoint'] = 'day 10'
    synth_data_15d['timepoint'] = 'day 15'
    synth_data_total = pandas.concat([synth_data_0d, synth_data_5d, synth_data_10d, synth_data_15d])

    return synth_data_total


def search_space_prolif_invsig_kde_mse(all_obs, depth_range=(0.05, 0.95),
                                       loc_range=(0.5, 0.8), pitch_range=(4,12),
                                       mtCN_opts=(500, 1000, 2000, 5000),
                                       iter_num=5000, init_seed=5, weighted_mse=False):
    #get observed mean heteroplasmy values
    day_vals_obs = ['d0_lhon', 'd5_lhon', 'd10_lhon', 'd15_lhon']
    d0_lhon = all_obs.loc[all_obs['condition_no_rep'] == 'd0_lhon',
                          ['umi_count_for_filtering', 'LHON_het', 'SILENT_ONLY_het']].copy()
    d0_lhon['mtDNA_umi_scale'] = d0_lhon['umi_count_for_filtering']/d0_lhon['umi_count_for_filtering'].max()
    d0_lhon['wt_het'] = 1 - d0_lhon[['LHON_het', 'SILENT_ONLY_het']].sum(axis=1)
    rng = numpy.random.default_rng(seed=init_seed)
    search_res = {'depth':[],
                  'loc':[],
                  'pitch':[],
                  'mtCN':[],
                  'seed':[],
                  'day0_lhon_mean':[],
                  'day0_silent_mean':[],
                  'day5_lhon_mean':[],
                  'day5_silent_mean':[],
                  'day10_lhon_mean':[],
                  'day10_silent_mean':[],
                  'day15_lhon_mean':[],
                  'day15_silent_mean':[],
                  'kde_mse':[],
                  'mean_het_mse':[]}
    
    x_d = numpy.meshgrid(numpy.linspace(0, 1.0, 100), numpy.linspace(0, 1.0, 100))
    x_d = numpy.array([x_d[0].flatten(), x_d[1].flatten()]).T
    obs_kde_res = []
    for day_val in day_vals_obs[1:]:
        # instantiate and fit the KDE model
        kde = neighbors.KernelDensity(bandwidth=0.05, kernel='gaussian')
        kde.fit(all_obs.loc[all_obs['condition_no_rep'] == day_val, ['LHON_het', 'SILENT_ONLY_het']].values)
        # score_samples returns the log of the probability density
        obs_kde_res.append(numpy.exp(kde.score_samples(x_d)))
    obs_kde_res = numpy.vstack(obs_kde_res)
    obs_means = numpy.hstack([all_obs.groupby('condition_no_rep')['LHON_het'].mean()[day_vals_obs].to_numpy(),
                              all_obs.groupby('condition_no_rep')['SILENT_ONLY_het'].mean()[day_vals_obs].to_numpy()])
    
    xorder = ['day 0', 'day 5', 'day 10', 'day 15']
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

        #run the model
        iter_res = train_one_invsig_model(d0_lhon, depth=depthval, loc=locval, pitch=pitchval, binomial_count=mtCNval,
                                          start_seed=iter_idx)

        synth_kde_res = []
        for day_val in xorder[1:]:
            # instantiate and fit the KDE model
            kde = neighbors.KernelDensity(bandwidth=0.05, kernel='gaussian')
            kde.fit(iter_res.loc[iter_res['timepoint'] == day_val, ['LHON_het', 'SILENT_ONLY_het']].values)
            # score_samples returns the log of the probability density
            synth_kde_res.append(numpy.exp(kde.score_samples(x_d)))
        synth_kde_res = numpy.vstack(synth_kde_res)
        
        lhon_means = iter_res.groupby('timepoint')['LHON_het'].mean()[xorder]
        search_res['day0_lhon_mean'].append(lhon_means['day 0'])
        search_res['day5_lhon_mean'].append(lhon_means['day 5'])
        search_res['day10_lhon_mean'].append(lhon_means['day 10'])
        search_res['day15_lhon_mean'].append(lhon_means['day 15'])
        silent_means = iter_res.groupby('timepoint')['SILENT_ONLY_het'].mean()[xorder]
        search_res['day0_silent_mean'].append(silent_means['day 0'])
        search_res['day5_silent_mean'].append(silent_means['day 5'])
        search_res['day10_silent_mean'].append(silent_means['day 10'])
        search_res['day15_silent_mean'].append(silent_means['day 15'])

        kde_mse_val = numpy.mean((obs_kde_res - synth_kde_res)**2)
        search_res['kde_mse'].append(kde_mse_val)
        mean_het_mse_val = numpy.mean((numpy.hstack([lhon_means, silent_means]) - obs_means)**2)
        search_res['mean_het_mse'].append(mean_het_mse_val)
    return pandas.DataFrame(search_res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('heteroplasmy_df_csv')
    parser.add_argument('outdir')
    parser.add_argument('--iter_num', type=int, help='Number of iterations to run before checkpointing hyperparameter search to a file. [default: %(default)d]', default=500)
    parser.add_argument('--iter_total', type=int, help='Total number of models to train in this hyperparameter search. [default: %(default)d]', default=8000)
    parser.add_argument('--iter_count', type=int, help='The iteration number with which to start the hyperparameter search. Used in case the search is interrupted and needs to be restarted from the previous checkpoint. [default: %(default)d]', default=0)
    args = parser.parse_args()

    #get single cell heteroplasmy estimates
    scilite_estimates = pandas.read_csv(args.heteroplasmy_df_csv)
    scilite_estimates.columns = scilite_estimates.columns.str.replace('_HET', '_het')
    scilite_estimates['condition_no_rep'] = scilite_estimates['condition'].str.split('_r', expand=True)[0]

    tp_map = {'d0':'day 0', 'd5':'day 5', 'd10':'day 10', 'd15':'day 15'}
    scilite_estimates['timepoint'] = [tp_map[elt.split('_')[0]] for elt in scilite_estimates['condition_no_rep'].to_numpy()]

    day_vals = ['d0_lhon', 'd5_lhon', 'd10_lhon', 'd15_lhon']
    lhon_cells = scilite_estimates[scilite_estimates['condition_no_rep'].isin(day_vals)].copy()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    iter_num = args.iter_num
    iter_count = args.iter_count
    iter_res_paths = []
    while iter_count < args.iter_total:
        test = search_space_prolif_invsig_kde_mse(lhon_cells, iter_num=iter_num, init_seed=iter_count,
                                                  depth_range=(0.01, 0.99),
                                                  loc_range=(0.4, 0.9), pitch_range=(1,13),
                                                  mtCN_opts=(500, 1000, 2000, 4000))
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
