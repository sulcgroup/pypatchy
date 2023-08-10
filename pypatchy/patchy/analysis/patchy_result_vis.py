import pandas as pd
import itertools
import altair as alt
import math


def showPatchyResults(results, target_name, plot_relative=True):
    target = results.targets[target_name]
    data = []
    for run_result in results.flat_runs():
        data += results.analyseClusterYield(run_result, target=target_name)
    df = pd.DataFrame(data)

    rule_data = pd.DataFrame(
        [
            {
                'run_name': results.runs[int(i / len(results.particle_set))].param_name,
                'cube_name': results.particle_set[i % len(results.particle_set)].param_name,
                'level': lvl
            }
            for i, lvl in enumerate(itertools.chain.from_iterable([r.cube_type_levels for r in results.runs]))
        ]
    )

    df['time'] /= 10e6  # to make data more readable

    df_reduced = df.loc[:, ("shape", "temp", "potential", "time", "num_assemblies", "duplicate")]
    df_reduced = df_reduced.drop_duplicates()
    grouped_data = df.groupby(["shape", "temp", "potential", "time", "duplicate"], as_index=False)
    df_reduced['yield_avg'] = grouped_data['yield'].mean()['yield']
    df_reduced['yield_mins'] = grouped_data['yield'].min()['yield']
    df_reduced['yield_maxs'] = grouped_data['yield'].max()['yield']

    yield_data = [df_reduced[df_reduced['temp'] == t] for t in results.temperatures()]
    for t, yld in zip(results.temperatures(), yield_data):
        yld['yield_relative'] = yld.loc[:, 'yield_avg'] / (yld.loc[:, 'num_assemblies'] * target['rel_count'])

    chart = alt.hconcat(
        alt.Chart(rule_data).mark_bar().encode(
            x=alt.X('run_name', title="Configuration"),
            y=alt.Y('level', title="Level"),
            color=alt.Color('cube_name', title="Type")
        ),
        *[
            alt.Chart(yld, title="Assembly Yields at T=%f" % results.temperatures()[i]).mark_line().encode(
                x=alt.X('time:Q', title="Time (megasteps)", axis=alt.Axis(tickCount=5)),
                y=alt.Y(
                    f"{'yield_relative' if plot_relative else 'yield'}:Q",
                    title="Yield",
                    scale=alt.Scale(
                        domain=(
                            0,
                            1.0 if plot_relative else math.ceil((yld['num_assemblies'] * target['rel_count']).max())
                            # math.max(math.ceil(df['yield'] / (df['num_assemblies'] * target['rel_count'])))
                        )
                    )
                ),
                color=alt.Color('shape', scale=alt.Scale(scheme="accent"), title="Configuration")
            ).properties(width=int(800 / len(results.temperatures())))
            for i, yld in enumerate(yield_data)
        ]
    )

    # ) + alt.Chart(df).mark_errorband(extent='ci', opacity=0.2).encode(
    #     x=alt.X('time:Q'),
    #     y=alt.Y('yield', title='Yield'),
    #     color = alt.Color('shape', scale=alt.Scale())
    # )).properties(width=600, height=450).facet(column='potential', row='type').properties(title=sim_name)

    chart.properties(title=f"{results.export_name} - {target_name}")
    # chart.save(sims_root() + os.sep + results.export_name + os.sep + results.export_name + ".html")
    return chart


def showClusterSizes(results, targets_for_reference):
    data = []
    for run_result in results.flat_runs():
        data += results.getClusterSizeData(run_result)
    df = pd.DataFrame(data)
    df['time'] /= 10e6  # to make data more readable

    max_cluster_size = math.ceil((df['cluster_size']).max())
    max_cluster_size = max(max_cluster_size,
                           *[len(results.targets[target_name]['graph']) for target_name in targets_for_reference])

    data_by_temperature = [df[df['temp'] == t] for t in results.temperatures()]

    sub_charts = [0 for _ in data_by_temperature]
    for tidx, tdata in enumerate(data_by_temperature):
        c = [
            alt.Chart(tdata[tdata['i'] == cluster_idx],
                      title="Assembly Yields at T=%f" % results.temperatures()[tidx]).mark_point().encode(
                x=alt.X('time:Q', title="Time (megasteps)", axis=alt.Axis(tickCount=5)),
                y=alt.Y(
                    "cluster_size:Q",
                    title="Cluster Size",
                    scale=alt.Scale(
                        domain=(
                            0,
                            max_cluster_size
                        )
                    )
                ),
                color=alt.Color('shape', scale=alt.Scale(scheme="accent"), title="Configuration")
            ).properties(width=int(800 / len(results.temperatures())))
            for cluster_idx in range(tdata['i'].min(), tdata['i'].max())
        ]
        if len(sub_charts) == 0:
            continue
        sub_charts[tidx] = c[0]
        for i in range(1, len(c)):
            sub_charts[tidx] += c[i]
        for target_name in targets_for_reference:
            line = alt.Chart(
                pd.DataFrame({'cluster_size': [len(results.targets[target_name]['graph'])]})).mark_rule().encode(
                y='cluster_size')
            sub_charts[tidx] += line

    chart = alt.hconcat(*sub_charts)

    return chart
