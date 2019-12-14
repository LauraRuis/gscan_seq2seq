import argparse
import datetime
import itertools
import os
import submitit
from pprint import pprint

from seq2seq.__main__ import main as runner_main
from seq2seq.__main__ import parser as runner_parser

os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--dry', action='store_true')
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--local', action='store_true')


# key => k; some_key => sk
def make_prefix(key):
    tokens = key.split('_')
    return ''.join(w[0] for w in tokens)


def expand_args(params, runs):
    sweep_args = {k: v for k, v in params.items() if isinstance(v, list)}
    # sweep :: [{arg1: val1, arg2: val1}, {arg1: val2, arg2: val2}, ...]
    sweep = [
        dict(zip(sweep_args.keys(), vs))
        for vs in itertools.product(*sweep_args.values())
    ]
    expanded = []
    for swargs in sweep:
        for n in range(runs):
            new_args = {**params, **swargs}  # shallow merge
            # new_args['xpid'] = '{}--{:02d}'.format('-'.join(
            #     [f'{make_prefix(k)}{v}' for k, v in swargs.items()]), n)
            expanded.append(new_args)
    return expanded


# NOTE for list arguments, use string.
args_grid = dict(
    mode=["train"],
    data_directory=["data/full_uniform_nononce"],
    attention_type=["bahdanau", "luong"],
    output_directory=["uniform_small_model"],
    training_batch_size=[200, 300, 500]
)


# NOTE params is a shallow merge, so do not reuse values
def make_command(params, unique_id):
    params['output_directory'] = ('/checkpoint/lauraruis/%s/gscan-random-split-%s' %
                                  (datetime.date.today().strftime('%y-%m-%d'),
                                   unique_id))
    # creating cmd-like params
    params = itertools.chain(*[('--%s' % k, str(v))
                               for k, v in params.items()])
    return list(params)


args = parser.parse_args()
args.runs = 1  # TODO: HARDCODED - make it flexible
args_grid = expand_args(args_grid, args.runs)
print(f"Submitting {len(args_grid)} jobs to Slurm...")

uid = datetime.datetime.now().strftime('%H-%M-%S-%f')
job_index = 0

for run_args in args_grid:
    job_index += 1
    flags = runner_parser.parse_args(make_command(run_args, uid))

    if args.local:
        executor_cls = submitit.LocalExecutor
    else:
        executor_cls = submitit.SlurmExecutor

    executor = executor_cls(folder='/checkpoint/lauraruis/gscan/jobs')

    executor.update_parameters(
        # slurm setup
        partition='learnfair',
        time=10,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name='gscan-random-split',
        mem="32GB",
        cpus_per_task=10,
        num_gpus=1,
    )
    print('########## Job {:>4}/{} ##########\nFlags: {}'.format(
        job_index, len(args_grid), flags))

    if not args.dry:
        print('Sending to slurm... ', end='')
        job = executor.submit(runner_main, flags)
        print('Submitted with job id: ', job.job_id)
    print('Log directory: ', flags.output_directory)

    if args.local:
        print('Only running one job on devfair for debugging...')
        pprint(args)
        import sys
        sys.exit(0)
