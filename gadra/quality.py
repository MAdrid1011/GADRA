"""Post-execution diagnostics from committed output depths."""
import numpy as np


def at_committed_depths(observations,stops,config):
    stops=np.asarray(stops)
    scores=np.asarray(observations['scores'])
    if (scores.ndim!=3 or stops.dtype.kind not in 'iu' or
        stops.shape!=(scores.shape[1],) or not len(stops) or
        np.any(stops<0) or np.any(stops>=scores.shape[0])):
        raise ValueError('Quality requires one valid committed integer depth per position')
    positions=np.arange(len(stops))
    def output(name):
        values=observations['output_'+name] if 'output_'+name in observations else observations[name]
        return values[stops,positions]
    chosen=output('winner')
    regret=output('regret')
    tolerance=config.get('quality_regret_tolerance',.01)
    agreement=float(np.mean(chosen==observations['target']))
    within=float(np.mean(regret<=tolerance))
    return dict(class_agreement=agreement,within_tolerance=within,
        mean_cosine=float(output('cosine').mean()),
        rmse=float(np.sqrt(output('squared_error').mean())),
        mean_final_score_regret=float(regret.mean()),max_final_score_regret=float(regret.max()),
        exited_fraction=float(np.mean(stops<len(observations['scores'])-1)),
        mean_completed_checkpoints=float((stops+1).mean()),
        stop_histogram=np.bincount(stops,minlength=len(observations['scores'])).tolist(),
        passes_thresholds=(agreement>=config['quality_min_class_agreement'] and
                           within>=config['quality_min_within_tolerance']),
        scope='Captured semantic readouts at online-committed depths')
