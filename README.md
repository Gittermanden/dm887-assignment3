# dm887-assignment3




## Usage


To access the program:
```
git clone https://github.com/Gittermanden/dm887-assignment3.git
cd dm887-assignment3
conda env create -f environment.yml
conda activate dm887-3
```
And you're in.

The main.py-file acts as a testsuite that allows for training a specified algorithms on a specified environments with a specified seeds. During training, the metrics that are tweaked, or whatever, are logged by tensorlog. 

Once an algorithm has completed its training, the findings are stored in the "./eval_results/{seed_number}/{environment}/{algorithm}"-folder, such as the best performing version of the model encountered during the learning process, as well as the complete log of evaluations throughout. 

After the training, a video of the best performing version of the model is recorded in its environment, and a graph of the model's learning rate. These can be found alongside the rest of the findings.

The learning rate: "./eval_results/{seed_number}/{environment}/{algorithm}/learning_curve.png".

The video: "./eval_results/{seed_number}/{environment}/{algorithm}/best_mode_ {environment}_{\...}.mp4".

If multiple algorithms have been run on the same environment, a comparison graph is generated, showcasing how each algorithm performed in the given environment, compared. This graph can be found in at "./eval_results/{seed_number}/{environment}/comparison_{environment}.png".



To run the testsuite, you can use the command:
```
python -m ./main.py
```
To configure the tests to your liking is readily apparent (So don't go bothering me... =) \).
If there is a parameter you'd like to change, you'll find it in either the main.py- or evaluator.py-files. 

Use this command for big uppies (Generates detailed graphs from the tensorlog using tensorboard): 
```
uvx --with setuptools==81.0.0 tensorboard --logdir ./tb_logs/
```

## Info
The CarRacing-v3 is by greyscaled and resized to aid performance. This should allow SAT and TD3 to use the default buffer size.


PPO is straight up wicked. It'll often overshoot the total amount of steps set to take. And if the total amount is small enough, the additional amount of iterations can become quite egregious.

This is happens because the algorithm has to run the # of parallel environments times value of n_steps each evaluation. (For the base values, this is: 8*2048, meaning that no matter what # of steps it is supposed to take, it'll only stop at steps in base 16192, or something).

All the algorithms will by default train on 8 environments in parallel. This isn't optimal for SAC and TD3 (debatable) but performance. If you're bothered by that, feel free to change it and wait an eternity for it to complete. 

If some of the contents don't work for you, major skill issue tbh...