import importlib
import pandas as pd
import matplotlib.pylab as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
import numpy as np
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api
import statsmodels.stats
import itertools

def color_violinplot(parts, c='black'):
    for pc in parts['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(None)
    parts['cmeans'].set_color(c)
    parts['cmaxes'].set_color(c)
    parts['cmins'].set_color(c)
    parts['cbars'].set_color(c)

if __name__=='__main__':
    threshold_angle_quantile = .9
    threshold_n_trials = 1
    threshold_lik = .75
    threshold_over_angle_trials = 10
    target_participants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    target_sessions = [0, 1, 2, 3, 4, 5, 6]

    df_all = pd.read_csv('bhv.csv', index_col=0)

    threshold_angle = df_all['angle_diff'].quantile(threshold_angle_quantile)
    print('Outliers > {:.4f} deg'.format(threshold_angle))

    for ss in df_all.session.unique():
        if len(df_all.query('angle_diff > {} & session == {}'.format(threshold_angle, ss))) > threshold_over_angle_trials:
            print('Session #{} rejected'.format(ss))
            df_all = df_all.query('session != {}'.format(ss))

    # ==============================
    # individual session
    # ==============================
    # for ss in df_all.session.unique():
    for ss in [82]:
        df = df_all.query('session == {}'.format(ss))

        bandit_labels = ['K', 'W']
        bandit_names = ['Black', 'White']
        bandit_color = ['k', 'w']
        lss = ['-', '--']

        plt.figure(1); plt.clf()
        plt.subplot(2, 1, 1)
        kt = np.zeros(len(df))
        wt = np.zeros(len(df))
        for tt in range(len(df)):
            plt.plot([tt, tt], [0, df['angle_diff'][tt]], ls='-', c='k')
            if df['angle_keep'][tt] == 1:
                plt.plot(tt, df['angle_diff'][tt], 'o', c='w', mec='k', ms=9)
            if df['target_dot'][tt] == 0:
                # lk = plt.plot(tt, df['angle_diff'][tt], 'o', c='k', mec='k')
                kt[tt] = df['angle_diff'][tt]
                wt[tt] = np.nan
            else:
                # lw = plt.plot(tt, df['angle_diff'][tt], 'o', c='w', mec='k')
                wt[tt] = df['angle_diff'][tt]
                kt[tt] = np.nan
        plt.plot(range(len(df)), kt, 'o', c='k', mec='k', label='Black (K)')
        plt.plot(range(len(df)), wt, 'o', c='w', mec='k', label='White (W)')
        plt.ylim([0, threshold_angle])
        plt.legend(loc='upper right')
        plt.xlabel('Trial')
        plt.ylabel('MDEE [deg]')
        plt.savefig('figs/session/dir_p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

        plt.figure(2); plt.clf()
        for bb in range(2):
            plt.subplot(2, 1, bb + 1)
            label_u ='$\Omega_' r'{\rm ' + '{}'.format(bandit_labels[bb]) + '}$'
            lu = plt.plot(df['rew_max{}'.format(bb)], c='k', ls='--', label=label_u)
            label_q ='$Q_' r'{\rm ' + '{}'.format(bandit_labels[bb]) + '}$'
            lq = plt.plot(df['q{}'.format(bb)], ls='-', c='k', label=label_q)
            for tt in range(len(df)):
                if df['target_dot'][tt] == bb:
                    bp = plt.plot(tt, df['rew'][tt], 'o', c=bandit_color[bb], mec='k')
            bp[0].set_label('Rew')
            plt.ylim([-1, 11])
            plt.ylabel('{} arm'.format(bandit_names[bb]))
            plt.legend(loc='upper right')
            if bb < 1:
                plt.xticks([])
            else:
                plt.xlabel('Trial')
        plt.savefig('figs/session/uqrew_p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

        plt.figure(3); plt.clf()
        ax = plt.subplot(2, 1, 1)
        for tt in range(len(df)):
            if df['target_lik'][tt] < threshold_lik:
                ax.axvspan(tt - .5, tt + .5, color='gray', ec=None, alpha=.5)
        for bb in range(2):
            label_p ='$P(' r'{\rm ' + '{}'.format(bandit_labels[bb]) + '})$'
            lp = plt.plot(df['lik{}'.format(bb)], c='k', ls=lss[bb], label=label_p)
        plt.ylim([-.1, 1.1])
        plt.legend(loc='upper right')
        plt.ylabel('Likelihood')
        plt.xlabel('Trial')
        plt.savefig('figs/session/lk_p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

        plt.figure(4); plt.clf()
        plt.subplot(311)
        l0 = plt.plot(df['rew_max0'], ls='--', label='r0')
        l1 = plt.plot(df['rew_max1'], ls='--', label='r1')
        for tt in range(len(df)):
            if df['target_dot'][tt] == 0:
                c = l0[0].get_color()
            else:
                c = l1[0].get_color()
            plt.plot(tt, df['rew'][tt], 'o', c=c, mec='k')

        plt.subplot(311)
        plt.plot(df['q0'], c=l0[0].get_color(), ls='-', label='q0')
        plt.plot(df['q1'], c=l1[0].get_color(), ls='-', label='q1')
        plt.xlim([-1, len(df)])
        plt.xticks([])
        plt.ylabel('Reward')
        plt.legend(loc='upper right')

        plt.subplot(312)
        plt.plot(df['lik0'])
        plt.plot(df['lik1'])
        for tt in range(len(df)):
            if df['target_dot'][tt] == 0:
                c = l0[0].get_color()
            else:
                c = l1[0].get_color()
            if df['target_change'][tt] == 1:
                mec = 'k'
            else:
                mec = None
            if df['angle_keep'][tt] == 1:
                marker = 's'
            else:
                marker = 'o'
            plt.plot(tt, df['lik{}'.format(df['target_dot'][tt])][tt], c=c, mec=mec, marker=marker)
        plt.plot([0, len(df)], [threshold_lik, threshold_lik], 'k--')
        plt.plot([0, len(df)], [1 - threshold_lik, 1 - threshold_lik], 'k--')
        plt.xlim([-1, len(df)])
        plt.xticks([])
        plt.ylabel('Likelihood')
        # plt.xlabel('Trial')

        plt.subplot(313)
        for tt in range(len(df)):
            if df['target_dot'][tt] == 0:
                c = l0[0].get_color()
            else:
                c = l1[0].get_color()
            if df['target_change'][tt] == 1:
                mec = 'k'
            else:
                mec = None
            if df['angle_keep'][tt] == 1:
                marker = 's'
            else:
                marker = 'o'
            plt.plot(tt, df['angle_diff'][tt], c=c, mec=mec, marker=marker)
            plt.plot([tt, tt], [0, df['angle_diff'][tt]], c=c)
        plt.xlim([-1, len(df)])
        # plt.xticks([])
        plt.ylim([0, threshold_angle])
        plt.ylabel('Angle error')
        plt.xlabel('Trial')

        plt.pause(.1)
        plt.savefig('figs/session/p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

    # ==============================
    # grand analysis
    # ==============================
    sess_queries = 'participant in {} & session_within_participant in {}'.format(target_participants, target_sessions)
    df_all0 = df_all.query(sess_queries)
    dir_err = df_all0['angle_diff']
    plt.figure(11).clf()
    plt.hist(dir_err, bins=100, color='k')
    ylim = plt.gca().get_ylim()
    plt.plot([threshold_angle, threshold_angle], ylim, 'k--')
    plt.plot(threshold_angle, ylim[1], 'ko')
    plt.plot([dir_err.mean(), dir_err.mean()], ylim, 'k--')
    plt.plot(dir_err.mean(), ylim[1], 'ks')
    plt.plot([dir_err.median(), dir_err.median()], ylim, 'k--')
    plt.plot(dir_err.median(), ylim[1], 'kv')
    plt.xlabel('MDEE [deg]')
    plt.ylabel('Number of trials')
    plt.savefig('figs/angle_diff.pdf', bbox_inches='tight', transparent=True)
    print('\n=========================')
    print('Mean dir error:\t{}+-{}'.format(dir_err.mean(), dir_err.std()))

    # ==============================
    base_queries = 'angle_diff <= {} & participant in {} & session_within_participant in {}'.format(
            threshold_angle, target_participants, target_sessions)
    df_all1 = df_all.query(base_queries)
    # ==============================

    df0 = df_all1[df_all1['target_dot'] == 0]
    rew_diff0 = df0['rew_max0'] - df0['rew_max1']
    n_bins = 36
    plt.figure(1).clf()
    b0 = plt.hist(rew_diff0, bins=n_bins, color='k')
    df1 = df_all1[df_all1['target_dot'] == 1]
    rew_diff1 = df1['rew_max0'] - df1['rew_max1']
    plt.figure(1).clf()
    b1 = plt.hist(rew_diff1, bins=b0[1], color='k')

    plt.figure(1).clf()
    l =  b0[1][1:] - .5 * (b0[1][-1] - b0[1][-2])
    plt.plot(l, b0[0] / (b0[0] + b1[0]) * 100, 'k--o', label='Black (K)', mfc='k', mec='k')
    plt.plot(l, b1[0] / (b0[0] + b1[0]) * 100, 'k-o', label='White (W)', mfc='w', mec='k')
    plt.xticks(np.arange(-9, 9.1, 3))
    plt.ylabel('Proportion of the target arm [\%]')
    plt.xlabel('Diff in max rewards, ' + r'$\Omega_{\rm K} - \Omega_{\rm W}$')
    plt.legend(loc='upper right')
    plt.savefig('figs/prop_action.pdf', bbox_inches='tight', transparent=True)

    bin_width = l[1] - l[0]
    center_bin_idx_range = [16, 19]
    print('\n=========================')
    print('Propotion of the target arm (K vs W)')
    for bin_idxs in [
            range(0, center_bin_idx_range[0]),
            range(center_bin_idx_range[0], center_bin_idx_range[1] + 1),
            range(center_bin_idx_range[1] + 1, n_bins)]:
        s = [b0[0][bin_idxs].sum(),  b1[0][bin_idxs].sum()]
        # s = [b0[0].sum(),  b1[0].sum()]
        res = scipy.stats.chisquare(s, [np.sum(s)/2, np.sum(s)/2])
        print('Range: {:.1f}--{:.1f}'.format(l[bin_idxs].min() - bin_width * .5, l[bin_idxs].max() + bin_width * .5))
        print('\tchi2(1):\t{}'.format(res.statistic))
        print('\tpvalue:\t{}'.format(res.pvalue))

    plt.figure(2).clf()
    rew = df_all1['rew']
    dir_err = df_all1['angle_diff']
    plt.plot(rew, dir_err, 'ko', alpha=.2)
    plt.ylim([0, threshold_angle])
    plt.ylabel('MDEE [deg]')
    plt.xlabel('Reward ' + r'$r$')
    # plt.xlabel('Diff in max rewards, ' + r'$Q_{\rm K} - Q_{\rm W}$')
    # plt.legend(loc='upper right')
    plt.savefig('figs/corr_err_rew.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.pearsonr(rew, dir_err)
    print('\n=========================')
    print('Correlation between reward and dir error')
    print('Corr({}):\t{}'.format(len(df_all1) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    plt.figure(2).clf()
    df0 = df_all1[df_all1['prior_rew'] >= 0]
    prior_rew = df0['prior_rew']
    dir_err = df0['angle_diff']
    plt.plot(prior_rew, dir_err, 'ko', alpha=.2)
    plt.ylim([0, threshold_angle])
    plt.ylabel('MDEE [deg]')
    plt.xlabel('Reward in the prior trial' + r'$r[n - 1]$')
    plt.savefig('figs/corr_err_prior_rew.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.pearsonr(prior_rew, dir_err)
    print('\n=========================')
    print('Correlation between prior reward and dir error')
    print('Corr({}):\t{}'.format(len(df0) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    plt.figure(3).clf()
    rew = pd.concat([
        df_all1[df_all1['target_dot'] == 0]['rew_max0'],
        df_all1[df_all1['target_dot'] == 1]['rew_max1'],
        ])
    dir_err = pd.concat([
        df_all1[df_all1['target_dot'] == 0]['angle_diff'],
        df_all1[df_all1['target_dot'] == 1]['angle_diff'],
        ])
    plt.plot(rew, dir_err, 'ko', alpha=.2)
    plt.ylim([0, threshold_angle])
    plt.ylabel('MDEE [deg]')
    plt.xlabel('Max reward on the target arm, ' + r'$\Omega_{s}$')
    plt.savefig('figs/corr_err_rew_target.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.pearsonr(rew, dir_err)
    print('\n=========================')
    print('Correlation between max reward on the target arm and dir error')
    print('Corr({}):\t{}'.format(len(df_all1) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    plt.figure(3).clf()
    rew_diff = np.abs(df_all1['rew_max0'] - df_all1['rew_max1'])
    dir_err = df_all1['angle_diff']
    plt.plot(rew_diff, dir_err, 'ko', alpha=.2)
    plt.ylim([0, threshold_angle])
    plt.ylabel('MDEE [deg]')
    plt.xlabel('Diff in max rewards, ' + r'$|\Omega_{\rm K} - \Omega_{\rm W}|$')
    plt.savefig('figs/corr_err_rewdiff.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.pearsonr(rew_diff, dir_err)
    print('\n=========================')
    print('Correlation between diff in max rewards and dir error')
    print('Corr({}):\t{}'.format(len(df_all1) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    plt.figure(4).clf()
    plt.subplot(121)
    dir_err = df_all1['angle_diff']
    dir_err_hld = dir_err[df_all1['angle_keep'] == 1]
    dir_err_nhl = dir_err[df_all1['angle_keep'] == 0]
    # plt.bar([0, 1], [dir_err_hld.mean(), dir_err_nhl.mean()], yerr=[dir_err_hld.std(), dir_err_nhl.std()], color='k', tick_label=['Hold', 'No-hold'], capsize=5)
    # plt.boxplot([dir_err_hld, dir_err_nhl], labels=['Hold', 'No-hold'], sym='')
    parts = plt.violinplot([dir_err_hld, dir_err_nhl], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Hold', 'No-hold'])
    plt.xlabel('Trial condition')
    plt.ylabel('MDEE [deg]')
    # plt.ylim([0, threshold_angle])
    plt.savefig('figs/dir_err_trial_condition.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_ind(dir_err_hld, dir_err_nhl)
    print('\n=========================')
    print('Diff in MDEE between the Hold and No-hold trials')
    print('t({}):\t{}'.format(len(df_all1) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    plt.figure(4).clf()
    plt.subplot(121)
    dir_err = df_all1['angle_diff']
    dir_err_tchd = dir_err[df_all1['target_change'] == 1.]
    dir_err_tnch = dir_err[df_all1['target_change'] == 0.]
    parts = plt.violinplot([dir_err_tchd, dir_err_tnch], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Diff', 'Same'])
    plt.xlabel('Target continuity')
    plt.ylabel('MDEE [deg]')
    # plt.ylim([0, threshold_angle])
    plt.savefig('figs/dir_err_target_change.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_ind(dir_err_tchd, dir_err_tnch)
    print('\n=========================')
    print('Diff in MDEE between the target change and same')
    print('t({}):\t{}'.format(len(df_all1) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    # ==============================
    # RL model analysis
    # ==============================
    base_queries = 'participant in {} & session_within_participant in {}'.format(target_participants, target_sessions)
    df_all2 = df_all.query(base_queries)
    # ==============================

    pks = {'alpha':r'$\alpha$', 'init_q':r'$\delta$', 'gamma':r'$\gamma$', 'beta':r'$\beta$'}
    print('\n=========================')
    print('RL parameters')
    for pp, param in enumerate(pks.keys()):
        plt.figure(1).clf()
        plt.subplot(1, 4, 1)
        print('{}:\t{:.4f}+-{:.4f}'.format(param, df_all2.loc[0][param].mean(), df_all2.loc[0][param].std()))
        # plt.bar([0], df_all2.loc[0][param].mean(), yerr=df_all2.loc[0][param].std(), color='k', tick_label=pks[param], capsize=5)
        # plt.boxplot(df_all2.loc[0][param], labels=[pks[param]], whis=1.5)
        parts = plt.violinplot(df_all2.loc[0][param], showmeans=True)
        color_violinplot(parts)
        plt.xticks([1], [pks[param]])
        plt.savefig('figs/rl_params_{}.pdf'.format(param), bbox_inches='tight', transparent=True)

    # ==============================
    base_queries = 'angle_diff <= {} & participant in {} & session_within_participant in {}'.format( threshold_angle, target_participants, target_sessions)
    df_all3 = df_all.query(base_queries)
    # ==============================

    plt.figure(2).clf()
    plt.subplot(121)
    vr = np.zeros([len(target_participants), len(target_sessions)], dtype=int)
    vi = np.zeros_like(vr)
    for pp, p in enumerate(target_participants):
        df_pp = df_all3[df_all3['participant'] == p]
        for ss, s in enumerate(target_sessions):
            df_ss = df_pp[df_pp['session_within_participant'] == s]
            vr[pp, ss] = len(df_ss[df_ss['target_lik'] < threshold_lik]) # Exploration
            vi[pp, ss] = len(df_ss[df_ss['target_lik'] >= threshold_lik]) # Exploitation
    parts = plt.violinplot([vr.ravel(), vi.ravel()], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Explor', 'Exploi'])
    plt.ylabel('Number of trials')
    plt.xlabel('Phase')
    plt.savefig('figs/n_trials_phase.pdf', bbox_inches='tight', transparent=True)
    tres = scipy.stats.ttest_rel(vr.ravel(), vi.ravel())
    print('\n=========================')
    print('#trials for each phase')
    print('{}: {:.1f} ({}) vs {}: {:.1f} ({}) | t({}) = {:.3f}, p = {:.3f}'.format('Explor', np.mean(vr), len(vr.ravel()), 'Exploi', np.mean(vi), len(vi.ravel()), tres.df, tres.statistic, tres.pvalue))

    plt.figure(3).clf()
    plt.subplot(121)
    vr = np.zeros([len(target_participants), len(target_sessions)], dtype=float)
    vi = np.zeros_like(vr)
    for pp, p in enumerate(target_participants):
        df_pp = df_all3[df_all3['participant'] == p]
        for ss, s in enumerate(target_sessions):
            df_ss = df_pp[df_pp['session_within_participant'] == s]
            vr[pp, ss] = df_ss[df_ss['target_lik'] < threshold_lik]['angle_diff'].mean() # Exploration
            vi[pp, ss] = df_ss[df_ss['target_lik'] >= threshold_lik]['angle_diff'].mean() # Exploitation
    nan_k = np.logical_or(np.isnan(vr), np.isnan(vi))
    vr = vr[np.logical_not(nan_k)]
    vi = vi[np.logical_not(nan_k)]
    parts = plt.violinplot([vr.ravel(), vi.ravel()], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Explor', 'Exploi'])
    plt.ylabel('MDEE [deg]')
    plt.xlabel('Phase')
    plt.savefig('figs/dir_err_phase.pdf', bbox_inches='tight', transparent=True)
    tres = scipy.stats.ttest_rel(vr.ravel(), vi.ravel())
    print('\n=========================')
    print('Dir err for each phase')
    print('{}: {:.1f} ({}) vs {}: {:.1f} ({}) | t({}) = {:.3f}, p = {:.3f}'.format('Explor', np.mean(vr), len(vr.ravel()), 'Exploi', np.mean(vi), len(vi.ravel()), tres.df, tres.statistic, tres.pvalue))



    # ==============================
    # Conditioned analysis
    # ==============================
    df_all = pd.read_csv('bhv.csv', index_col=0)
    base_queries = 'prior_angle_diff <= {} & angle_diff <= {} \
            & participant in {} & session_within_participant in {} \
            & target_lik < {} & target_change == 1'.format(
            threshold_angle, threshold_angle,
            target_participants, target_sessions,
            threshold_lik)
    df_all = df_all.query(base_queries)

    queries = {
            'RH': "prior_target_lik < {} & angle_keep == 1".format(threshold_lik),
            'IH': "prior_target_lik >= {} & angle_keep == 1".format(threshold_lik),
            'RN': "prior_target_lik < {} & angle_keep == 0".format(threshold_lik),
            'IN': "prior_target_lik >= {} & angle_keep == 0".format(threshold_lik),
            }
    dfk = pd.DataFrame(index=[], columns=[])
    for ss, session in enumerate(df_all.session.unique()):
        participant = df_all.query('session == {}'.format(session))['participant'].unique()[0]
        _dfk = pd.DataFrame(index=[], columns=[])
        for qq, condition in enumerate(queries.keys()):
            _df = df_all.query(queries[condition] + '& session == {}'.format(session))
            if len(_df) >= threshold_n_trials:
                _dfk0 = pd.DataFrame(
                        data = [{
                            'mdee': _df['angle_diff'].mean(),
                            'phase': condition[0],
                            'condition': condition[1],
                            'participant': participant,
                            'session': session,
                            'phasecond': condition[0] + condition[1],
                            'lik': _df['target_lik'].mean(),
                            }])
                _dfk = pd.concat([_dfk, _dfk0], axis=0)
        if len(_dfk) == len(queries):
            dfk = pd.concat([dfk, _dfk], axis=0)
    dfk.reset_index()
    n_participants = len(dfk['participant'].unique())
    n_sessions = len(dfk['session'].unique())
    print('#participant: {} ({} removed)'.format(n_participants, len(df_all['participant'].unique()) - n_participants))
    print('#session: {} ({} removed)'.format(n_sessions, len(df_all['session'].unique()) - n_sessions))

    # =================================
    # Test MDEE for state
    # =================================
    print('\n=========================')
    print('MDEE at state')
    f = 'mdee ~ phase + condition + C(participant) + C(session) + phase*condition'
    model = statsmodels.formula.api.ols(f, dfk).fit()
    # print(model.summary())
    aov_table = sm.stats.anova_lm(model, type=2)
    print(aov_table)

    for c0, c1 in itertools.combinations(queries.keys(), 2):
        mdee0 = dfk.query("phase == '{}' & condition == '{}'".format(c0[0], c0[1]))['mdee']
        mdee1 = dfk.query("phase == '{}' & condition == '{}'".format(c1[0], c1[1]))['mdee']
        tres = scipy.stats.ttest_rel(mdee0, mdee1, alternative='two-sided')
        print('{} vs {}: {}'.format(c0, c1, tres))

    plt.figure(0).clf()
    plt.subplot(121)
    vals = np.zeros([n_sessions, len(queries)])
    for qq, condition in enumerate(queries.keys()):
        vals[:, qq] = dfk.query("phase == '{}' & condition == '{}'".format(condition[0], condition[1]))['mdee']
        plt.plot(qq + 1 + np.random.uniform(-.2, .2, size=n_sessions), vals[:, qq], 'ko', alpha=.2)
    vparts = plt.violinplot(vals, showmeans=True)
    color_violinplot(vparts)
    plt.xticks(np.arange(len(queries)) + 1, queries.keys())
    plt.ylabel('MDEE [deg]')
    plt.xlabel('State')
    plt.savefig('figs/mdee_states.pdf'.format(), bbox_inches='tight', transparent=True)

    # =================================
    # Test likelihood for state
    # =================================
    print('\n=========================')
    print('Likelihood at state')
    f = 'lik ~ phase + condition + C(participant) + C(session) + phase*condition'
    model = statsmodels.formula.api.ols(f, dfk).fit()
    aov_table = sm.stats.anova_lm(model, type=2)
    print(aov_table)

    plt.figure(1).clf()
    plt.subplot(121)
    vals = np.zeros([n_sessions, len(queries)])
    for qq, condition in enumerate(queries.keys()):
        vals[:, qq] = dfk.query("phase == '{}' & condition == '{}'".format(condition[0], condition[1]))['lik']
        plt.plot(qq + 1 + np.random.uniform(-.2, .2, size=n_sessions), vals[:, qq], 'ko', alpha=.2)
    vparts = plt.violinplot(vals, showmeans=True)
    color_violinplot(vparts)
    plt.xticks(np.arange(len(queries)) + 1, queries.keys())
    plt.ylabel('Likelihood')
    plt.xlabel('State')
    plt.savefig('figs/lik_states.pdf'.format(), bbox_inches='tight', transparent=True)

    # =================================
    # Test likelihood x MDEE
    # =================================
    print('\n=========================')
    print('Correlation between MDEE and likelihood')
    res = scipy.stats.pearsonr(dfk['lik'], dfk['mdee'])
    print('Corr(df: {}):\t{:.3f}'.format(len(dfk) - 2, res[0]))
    print('pvalue:\t{:.3f}'.format(res[1]))
    plt.figure(2).clf()
    plt.plot(dfk['lik'], dfk['mdee'], 'ko')
    rl = np.polyfit(dfk['lik'], dfk['mdee'], 1)
    y = np.poly1d(rl)([dfk['lik'].min(), dfk['lik'].max()])
    plt.plot([dfk['lik'].min(), dfk['lik'].max()], y, 'k')
    plt.xlabel('Likelihood')
    plt.ylabel('MDEE [deg]')
    plt.savefig('figs/lik_mdee.pdf'.format(), bbox_inches='tight', transparent=True)
    
    plt.show()
