import torch as torch
import inspect
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import types as types
from tqdm import tqdm
import collections
import copy

#@title Game Solvers { display-mode: "code" }
## DESCRIPTION: THIS IS WHERE YOU PUT GAME SOLVER METHODS, INSIDE THIS CLASS

class Game:
    ## DESCRIPTION OF gradDescent: VERY SHITTY METHOD, just SIMULTANOUS STEP OPTIMIZATIONS OF STRATEGIES FOR ALL PLAYERS

    ## ASSUMES ALL PLAYERS HAVE TORCH TENSORS FOR STRATEGIES
    def gradDescent(self, start_strategies, optimizers, max_epochs= 1000, noisy = False, epsilon = .1, epsilon_spread = 10):
        strategy_parameters = []
        for strategy in start_strategies:
            if torch.is_tensor(strategy):
                strategy_parameters.append([strategy])
            elif isinstance(strategy, Player):
                strategy_parameters.append(strategy.get_params())
            elif isinstance(strategy, nn.Module):
                strategy_parameters.append(strategy.parameters())
            else:
                raise Exception("One of these starting strategies was not an Class or a Tensor Yo")
        ##start_strategies must have grad enabled
        ##This is to allow you to input just one optimizer
        if not isinstance(optimizers, list):
            optimizers = [optimizers for _ in range(len(self.players))]
        ##Initialize some Optimizers
        optim = [optimizers[i](strategy_parameters[i]) for i in range(len(self.players))]

        def isolated_step(tuplet):
            ###loss, optim, param = tuplet[0], tuplet[1], tuplet[2]
            grad_ = torch.autograd.grad(tuplet[0], tuplet[2], retain_graph = True, allow_unused=True)
            back = torch.autograd.backward(tuplet[2], grad_, retain_graph = True)
            # tuplet[1].step()

        #RUn THem
        for i in tqdm(range(max_epochs)):
        # for i in range(max_epochs):
            list(map(lambda x:x.zero_grad(),optim))
            # if i % 100 == 0:
            #     # print(rewards)
            #     rewards = self.play(start_strategies)
            #     (-rewards[0]).backward()
            #     print(start_strategies[0].grad)
            #     list(map(lambda x:x.zero_grad(),optim))

            if noisy == False:
                rewards = self.play(start_strategies)
            else:
                rewards = self.noisy_play(start_strategies, epsilon, noisy, [epsilon_spread for I in range(len(start_strategies))])



            # print(rewards)
            loss = [-x for x in rewards]

            list(map(isolated_step, zip(loss,optim,strategy_parameters)))
            # if i % 100 == 0:
            #     print(start_strategies[0].grad)
            list(map(lambda x: x.step(), optim ))

            # for j in range(1):
            #     print(j)
            #     list(map(lambda x:x.zero_grad(),optim))
            #     loss = [-x for x in self.play(start_strategies)]
            #     loss[j].backward(retain_graph=True)
            #     optim[j].step()

            # for j in range(len(strategy_parameters)):
            #     list(map(lambda x:x.zero_grad(),optim))
            #     loss = [-x for x in self.play(start_strategies)]
            #     isolated_step((loss[j],optim[j],strategy_parameters[j]))

        return start_strategies, [-x for x in loss]

    def noisy_play(self, strats, epsilon, noise_array, epsilon_spread):
        new_strat_array = []

        class Noise_Strategy(Player):
            def __init__(self, old_strategy, noise):
                super(Noise_Strategy, self).__init__()
                self.old_strategy = old_strategy
                self.noise = noise
            def forward(self, x):
                no_noise_ans = self.old_strategy.forward(x)
                return no_noise_ans + (torch.rand(no_noise_ans.shape) - .5) * self.noise
            def get_params(self):
                return self.old_strategy.get_params



        class Random_Strategy(Player):
            def __init__(self, old_strategy, spread):
                super(Random_Strategy, self).__init__()
                self.old_strategy = old_strategy
                self.spread = spread
            def forward(self, x):
                no_noise_ans = self.old_strategy.forward(x)
                return (torch.rand(no_noise_ans.shape) - .5) * self.spread + (no_noise_ans - no_noise_ans)

            def get_params(self):
                return []


        for index, strategy in enumerate(strats):
            if torch.is_tensor(strategy):
                if np.random.rand() < epsilon:
                    new_strat_array.append((strategy - strategy) + (torch.rand(strategy.shape)-.5)* epsilon_spread[index])
                else:
                    new_strat_array.append(strategy + (torch.rand(strategy.shape)-.5)* noise_array[index])
            elif isinstance(strategy, Player):
                if np.random.rand() < epsilon:
                    new_strat_array.append(Random_Strategy(strategy, epsilon_spread[index]))
                else:
                    new_strat_array.append(Noise_Strategy(strategy, noise_array[index]))
            else:
                raise Exception("One of these starting strategies was not an Class or a Tensor Yo")



        return self.play(new_strat_array)
