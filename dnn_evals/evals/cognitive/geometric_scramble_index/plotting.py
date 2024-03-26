import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['plot_results']

def plot_results(results, subset_accuracy=True, chance=None):
    df = results['summary'] if 'summary' in results else results
    metric = 'correct_subset' if subset_accuracy else 'correct'

    def get_label(row):
        transform = row['transform']
        if transform=='none':
            return 'intact\n(none)'
        elif transform=='translate':
            return f'{row.dx},{row.dy}'
        elif transform=='rotate':
            return f'{row.angle}'
        elif transform=='scale':
            return f'x{row.scale}'
        elif transform=='scramble':
            return f'{row.block_dim}x{row.block_dim}'

    df_ = df.copy()
    df_['label'] = df_.apply(get_label, axis=1)
    N = len(df_[f'{metric}_mean'])

    # metric = "correct"
    sns.set_context("notebook", rc={"xtick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_, x="label", y=f'{metric}_mean', hue="transform", errorbar=None, ax=ax, dodge=False)
    ax.set_ylim([.0,1.1])
    ax.set_ylabel('proportion correct', labelpad=10, fontsize=20)
    ax.set_xlabel('transform params', labelpad=10, fontsize=20)
    if chance is not None:
        ax.axhline(chance, ls='--', color='black')

    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    x_coords = x_coords[0:N]
    y_coords = [p.get_height() for p in ax.patches]
    y_coords = y_coords[0:N]
    assert all(df_[f'{metric}_mean'] == y_coords), "Oops, trouble lining up your confidence intervals with the bars"

    lower_ci = df[f'{metric}_mean']-df[f"{metric}_lower_ci"]
    upper_ci = df[f"{metric}_upper_ci"]-df[f'{metric}_mean']
    err = [lower_ci, upper_ci]
    ax.errorbar(x=x_coords, y=y_coords, yerr=err, fmt="none", c="k")

    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.);
    sns.despine()
    
    return ax