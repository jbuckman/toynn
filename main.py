import torch, argparse, inspect
import numpy as np
import pandas as pd
from collections import defaultdict
from ipdb import launch_ipdb_on_exception
import matplotlib.pyplot as plt
import time, os

class defaultdotdict(defaultdict):
    def __getattr__(self, name): return self[name]
UNSPECIFIED = defaultdotdict(lambda:UNSPECIFIED)
def all_object_names(module): return {key for key, value in inspect.getmembers(module) if inspect.isfunction(object) or inspect.isclass(object)}
def fix_key(s): return "=".join([subs.replace('.','___') if i == 0 else subs for i, subs in enumerate(s.split("="))])
def subdictify(d):
    main = defaultdotdict(lambda:UNSPECIFIED, {key: value for key, value in d.items() if "___" not in key})
    sub = {key: subdictify({'___'.join(subkey.split("___")[1:]): value for subkey, value in d.items() if "___" in subkey and subkey.split("___")[0] == key}) for key in {key.split("___")[0] for key in d.keys() if "___" in key}}
    main.update(sub)
    return main
def load_to_dict(s_list): return subdictify(eval(f"dict({','.join([fix_key(s) for s in s_list])})"))
class do_periodically:
    def __init__(self, gap, slowdown_rate=0.):
        self.gap = gap
        self.slowdown_rate = slowdown_rate
        self.last = 0
        self.step = 0
    def __call__(self):
        self.step += 1
        if self.step >= self.last + self.gap:
            self.last = self.step
            self.gap += self.slowdown_rate
            return True
        else:
            return False
class do_time_freq:
    def __init__(self, time_gap):
        self.last_time = None
        self.time_gap = time_gap * 60 ## rate is given in hours
    def __call__(self):
        if self.last_time is None:
            self.last_time = time.time()
            return False
        elif time.time() >= self.last_time + self.time_gap:
            self.last_time = time.time()
            return True
        else:
            return False

def checkpoint(filename, model=None, x=None, ys=None):
    if filename != UNSPECIFIED:
        if not os.path.exists("out"): os.mkdir("out")
        if model is not None:
            print("Saving model as " + f"out/{filename}.pt")
            learner.nn.save(f"out/{filename}.pt")
        if x is not None:
            assert ys is not None
            print("Saving log as " + f"out/{filename}.pk")
            train_df = pd.DataFrame({"step": x, **ys})
            train_df.to_pickle(f"out/{filename}.pk")

if __name__ == '__main__':
    import tasks, algorithms

    parser = argparse.ArgumentParser()
    parser.add_argument("input", action='store', nargs='*')
    args = parser.parse_args().input
    print(f"Args: {args}")

    defaults = dict(minibatch_size=128,
                    task_class="mnist",
                    algo_class="vanilla",
                    gpu=True,
                    write_log_rate=10,
                    write_model_rate=60)

    with launch_ipdb_on_exception():
        args = load_to_dict(args)
        for key in defaults:
            if args[key] == UNSPECIFIED: args[key] = defaults[key]
        assert args.task_class in all_object_names(tasks)
        assert args.algo_class in all_object_names(algorithms)

        if torch.cuda.is_available() and args.gpu: device = torch.device('cuda:0')
        else: device = torch.device('cpu')

        if args.seed != UNSPECIFIED: torch.manual_seed(args.seed)

        task = getattr(tasks, args.task_class)(dataset_size=args.dataset_size, seed=args.seed, device=device, **args.task)
        learner = getattr(algorithms, args.algo_class)(task, device=device, dataset_size=args.dataset_size, **args.algo)

        should_dbg_print = do_time_freq(.1)
        should_eval_print = do_time_freq(1)
        should_write_model_checkpoint = do_time_freq(args.write_model_rate)
        should_write_log = do_time_freq(args.write_log_rate)
        should_eval = do_periodically(100, slowdown_rate=2)
        log_dirty = False

        train_logs_to_plot_x = []
        train_logs_to_plot_y = defaultdict(list)
        test_logs_to_plot_x = []
        test_logs_to_plot_y = defaultdict(list)
        losses = defaultdict(list)

        step = 0
        start_time = time.time()
        while args.duration == UNSPECIFIED or (time.time() - start_time) / 60 > args.duration:
            ## eval
            if (step <= 300 and step % 15 == 0) or should_eval():
                log_dirty = True
                ## test eval
                test_losses = defaultdict(list)
                test_gen = task.test_set_iterator(2*args.minibatch_size)
                for x_minibatch, y_minibatch, meta_minibatch in test_gen:
                    test_loss = learner.eval(step, x_minibatch, y_minibatch, meta_minibatch)
                    for k, v in test_loss.items(): test_losses[k] += v
                test_logs_to_plot_x.append(step)
                for name, vals in test_losses.items(): test_logs_to_plot_y[name].append(np.mean(vals))
                ## train eval
                train_gen = task.train_set_iterator(2*args.minibatch_size)
                train_losses = defaultdict(list)
                for x_minibatch, y_minibatch, meta_minibatch in train_gen:
                    train_loss = learner.eval(step, x_minibatch, y_minibatch, meta_minibatch)
                    for k, v in train_loss.items(): train_losses[k] += v
                if len(train_losses) > 0:
                    train_logs_to_plot_x.append(step)
                    for name, vals in train_losses.items(): train_logs_to_plot_y[name].append(np.mean(vals))
                else: ## if our data is IIID, train-set and test-set are indistinguishable
                    train_logs_to_plot_x = test_logs_to_plot_x
                    train_logs_to_plot_y = test_logs_to_plot_y
                if should_eval_print():
                    print(f"EVAL | step={step:6} | " +
                         (' '.join([f"train_{name}={np.mean(vals):.3}" for name,vals in train_losses.items()]) + " | " if len(train_losses) > 0 else "") +
                          ' '.join([f"test_{name}={np.mean(vals):.3}" for name,vals in test_losses.items()]), flush=True)


            ## write to disk
            if args.save and log_dirty and should_write_log():
                checkpoint(f'{args.save}.train', x=train_logs_to_plot_x, ys=train_logs_to_plot_y)
                checkpoint(f'{args.save}.test', x=test_logs_to_plot_x, ys=test_logs_to_plot_y)
            if should_write_model_checkpoint():
                checkpoint(args.save, model=learner.nn)

            ## perform learning
            x_minibatch, y_minibatch = task.train_sample(args.minibatch_size)
            loss = learner.learn(step, x_minibatch, y_minibatch)
            step += 1
            if should_dbg_print():
                print(f"DBG  | step={step:6} | " + ' '.join([f"{k}={v:.3}" for k, v in loss.items()]), flush=True)