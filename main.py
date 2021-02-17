import torch, argparse, inspect
import numpy as np
from collections import defaultdict
from ipdb import launch_ipdb_on_exception
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    import tasks, algorithms

    parser = argparse.ArgumentParser()
    parser.add_argument("input", action='store', nargs='*')
    args = parser.parse_args().input

    defaults = dict(steps=60000,
                    minibatch_size=128,
                    test_runs=100,
                    task_class="sin_1d",
                    algo_class="vanilla")

    with launch_ipdb_on_exception():
        args = load_to_dict(args)
        for key in defaults:
            if args[key] == UNSPECIFIED: args[key] = defaults[key]
        assert args.task_class in all_object_names(tasks)
        assert args.algo_class in all_object_names(algorithms)

        task = getattr(tasks, args.task_class)(dataset_size=args.dataset_size, **args.task)
        learner = getattr(algorithms, args.algo_class)(task, total_steps=args.steps, dataset_size=args.dataset_size, **args.algo)

        if args.seed != UNSPECIFIED: torch.manual_seed(args.seed)

        train_losses_to_plot_x = []
        train_losses_to_plot_y = []
        test_losses_to_plot_x = []
        test_losses_to_plot_y = []
        losses = defaultdict(list)
        for step in range(args.steps):
            x_minibatch, y_minibatch = task.train_sample(args.minibatch_size)
            loss = learner.learn(step, x_minibatch, y_minibatch)
            for k,v in loss.items(): losses[k].append(v)
            if (step + 1) % 100 == 0:
                print(f"{(step+1)/args.steps:.2%}\tstep={step+1}/{args.steps} | " + ' '.join([f"{k}={sum(v[-25:])/25:.3}" for k,v in losses.items()]))
                train_losses_to_plot_x.append(step)
                train_losses_to_plot_y.append(np.mean(losses['loss'][-100:]))
            if (step + 1) % 1000 == 0:
                print("Evaluating...")
                test_losses = defaultdict(list)
                test_gen = task.test_sample(args.minibatch_size)
                for test_step in range(args.test_runs):
                    try:
                        x_minibatch, y_minibatch = next(test_gen)
                        test_loss = learner.learn(step, x_minibatch, y_minibatch)
                        for k, v in test_loss.items(): test_losses[k].append(v)
                    except StopIteration:
                        pass
                for name, val in test_losses.items(): print(f"test {name}: {np.mean(val):.3}")
                test_losses_to_plot_x.append(step)
                test_losses_to_plot_y.append(np.mean(test_losses['loss']))

        print("Training complete.")
        if args.save != UNSPECIFIED:
            print("Saving as " + f"models/{args.save}.pt")
            with open(f"explore/models/{args.save}.pt", "wb") as f:
                torch.save(learner.nn, f)

        plt.plot(train_losses_to_plot_x, train_losses_to_plot_y, label="train")
        plt.plot(test_losses_to_plot_x, test_losses_to_plot_y, label="test")
        plt.legend()
        plt.show()

        if task.x_shape == [1]:
            train_xs = task.dataset
            train_ys = task.f(train_xs)
            eval_xs = torch.linspace(*task.test_range, 1400)[:,None]
            eval_ys = task.f(eval_xs)
            eval_guesses = learner.nn(eval_xs).detach()

            plt.scatter(train_xs, train_ys, label="train points")
            plt.plot(eval_xs, eval_ys, label="true function")
            plt.plot(eval_xs, eval_guesses, label="learned guesses")
            plt.legend()
            plt.show()

        input("Done. ")