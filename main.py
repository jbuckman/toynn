import torch, argparse, inspect
import numpy as np
from collections import defaultdict
from ipdb import launch_ipdb_on_exception
import matplotlib.pyplot as plt

def all_object_names(module): return {key for key, value in inspect.getmembers(module) if inspect.isfunction(object) or inspect.isclass(object)}
def load_to_dict(s): return eval(f"dict({s})")
import tasks, algorithms

parser = argparse.ArgumentParser()
parser.add_argument("--steps", default=60000, type=int)
parser.add_argument("--minibatch_size", default=128, type=int)
parser.add_argument("--dataset_size", default=None, type=int)
parser.add_argument("--test_runs", default=100, type=int)
parser.add_argument("--save", default=None)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--task", choices=all_object_names(tasks), default="sin_1d")
parser.add_argument("--algo", choices=all_object_names(algorithms), default="vanilla")
parser.add_argument("--task_params", type=load_to_dict, default="")
parser.add_argument("--algo_params", type=load_to_dict, default="")
args = parser.parse_args()

with launch_ipdb_on_exception():
    task = getattr(tasks, args.task)(dataset_size=args.dataset_size, **args.task_params)
    learner = getattr(algorithms, args.algo)(task, total_steps=args.steps, dataset_size=args.dataset_size, **args.algo_params)

    if args.seed is not None: torch.manual_seed(args.seed)

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
    if args.save is not None:
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

    if input("Inspect?"): 1+''