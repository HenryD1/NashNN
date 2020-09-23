from game_instances import *

num_nodes = 5
adj_mat = torch.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(i):
        if not ((i == 4) & (j == 0)):
            adj_mat[i,j] = 1

print(adj_mat)


num_workers = torch.sum(adj_mat).type(torch.int32)


class Network_Worker(Player):
    #manually implemented optimizer, need to include
    def __init__(self):
      super(Network_Worker, self).__init__()
      self.m1 = (2*torch.rand([10,num_workers + 1]) - 1)
      self.b1 = (2*torch.rand([10]) - 1)
      self.m2 = (2*torch.rand([1,10]) - 1)
      self.b2 = (2*torch.rand([1]) - 1)

      self.m1.requires_grad = True
      self.m2.requires_grad = True
      self.b1.requires_grad = True
      self.b2.requires_grad = True

    # x represents our data
    def forward(self, x):
      x1 = torch.relu((self.m1 @ x) + self.b1)
      output = (self.m2 @ x1 + self.b2)

      return output
    def get_params(self):
        return [self.m1, self.m2, self.b1, self.b2]





Network_game = NetworkPrincipalAgent(adj_mat)


strats = [(torch.ones((num_workers + 1 )))] + [Network_Worker() for _ in range(num_workers)]
strats[0].requires_grad = True

# Network_game.play(strats)[0].backward()

# print(strats[0].grad)


# strats = [Principal()] + [Network_Worker() for _ in range(num_workers)]

torch.autograd.set_detect_anomaly(True)

# print(Network_game.play(strats))

# strats = [(torch.ones((num_workers + 1 )))] + [torch.tensor(0. ,requires_grad=True) for _ in range(num_workers)]
# strats[0].requires_grad = True

# print(Network_game.play(strats))


Equilibrium = Network_game.gradDescent(strats, torch.optim.Adam, max_epochs=130000 , epsilon = 1, noisy = [.25 for _ in range(num_workers+ 1)] )



# print(Equilibrium[1])

# print(Equilibrium[1])

# Network_game = NetworkPrincipalAgent(adj_mat)

# print(Network_game.play(Equilibrium[0]))

# y = []
# x = torch.arange(100.) - 50
# for i in range(len(x)):
#     new_strats = Equilibrium[0].copy()
#     class Fake_worker(nn.Module):
#         def __init__(self):
#             super(Fake_worker, self).__init__()
#         def forward(self, x):
#             return

#     y.append(Network_game.play(new_strats))



# # plt.scatter(x,y)

#Print results about the Equilibrium

yo = torch.abs(Equilibrium[0][0])/torch.sum(torch.abs(Equilibrium[0][0]))

print(yo[0])
realized_adj_matrix = Network_game.adj_mat.detach().clone()
strat_index = 1
for i in range(5):
    for j in range(i):
        if not ((i == 4) & (j == 0)):
            realized_adj_matrix[i,j] = yo[strat_index]
            strat_index += 1

print(realized_adj_matrix)


print(Equilibrium[1])
effort = []
for strat in Equilibrium[0][1:]:
    effort.append(torch.sigmoid(strat.forward(yo)))

print(effort)
strat_index = 0
for i in range(5):
    for j in range(i):
        if not ((i == 4) & (j == 0)):
            realized_adj_matrix[i,j] = effort[strat_index]
            strat_index += 1

print(realized_adj_matrix)

print(Network_game.play(Equilibrium[0]))
print(Equilibrium[1])
Network_game.play(Equilibrium[0])[0].backward()

print(Equilibrium[0][0].grad)

print(Equilibrium[0][0])
