import game_definition
from game_definition import *


class FourByFour(Game):
    ## DESCRIPTION: YO JUST INPUT A 3 DIMENSIONAL TENSOR INTO THIS BI*TCH, with the first axis indicating which player gets what reward, second two axes are normal ass payoff matrix, then BOOm
    def __init__(self, payoff_mat):
        self.players = ["A", "B"]
        self.payoff_mat = payoff_mat
        self.function_players = []
    def play(self, strats):
        # strats2 = [nn.functional.softmax(strat(torch.tensor([1.]))) for strat in strats]
        strats2 = [nn.functional.softmax(strat) for strat in strats]

        # strats1 = [torch.relu(strat) for strat in strats]
        # strats2 = [strat/torch.sum(strat) for strat in strats1]
        return [torch.einsum('a,ac,c->', strats2[0], self.payoff_mat[i,:,:], strats2[1]) for i in [0,1]]

class FourByFourWNet(Game):

    ## DESCRIPTION: YO JUST INPUT A 3 DIMENSIONAL TENSOR INTO THIS BI*TCH, with the first axis indicating which player gets what reward, second two axes are normal ass payoff matrix, then BOOm
    def __init__(self, payoff_mat):
        self.players = ["A", "B"]
        self.payoff_mat = payoff_mat
        self.function_players = []
    def play(self, strats):
        strats2 = [nn.functional.softmax(strat.forward(torch.tensor([1.]))) for strat in strats]
        # strats2 = [nn.functional.softmax(strat) for strat in strats]

        # strats1 = [torch.relu(strat) for strat in strats]
        # strats2 = [strat/torch.sum(strat) for strat in strats1]
        return [torch.einsum('a,ac,c->', strats2[0], self.payoff_mat[i,:,:], strats2[1]) for i in [0,1]]


class NetworkPrincipalAgent(Game):
    def __init__(self, adj_mat):
        assert (adj_mat.shape[0] == adj_mat.shape[1] ) #this checks to see if the adjacency matrix is square.
        self.num_nodes = adj_mat.shape[0]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i <= j:
                    adj_mat[i,j] = 0
                else:
                    if adj_mat[i,j] != 0: #ok I see. We get a matrix with inputs already. And we set all nonzero inputs to 1 to clear for training.
                        adj_mat[i,j] = 1
        self.adj_mat = adj_mat
        self.num_agents = torch.sum(self.adj_mat).type(torch.int32) #Ok got it, each one is the number of agents.
        self.players = ["principal"] + ["agent" for _ in range(self.num_agents)]

    # def transmission_prob(self, prob_nodes, prob_edges):
    #     return (1 - torch.sum(1 - (prob_nodes * prob_edges)))

    def play(self, strats):

        distribution = nn.functional.softmax(strats[0], dim = 0)

        # print(distribution)

        effort = []

        # for j in range(1,len(strats)):
        #     effort.append(strats[j](distribution[j].reshape((1))))

        for j in range(1,len(strats)):
            effort.append(strats[j].forward(distribution))


        strat_index = 0
        realized_adj_matrix = self.adj_mat.detach().clone()
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if realized_adj_matrix[i,j] == 1:
                    realized_adj_matrix[i,j] = torch.sigmoid(effort[strat_index])
                    strat_index += 1

        # print(realized_adj_matrix)

        probability_vec = torch.zeros(1)
        probability_vec[0] = 1

        for node in range(1, num_nodes):
            prob_nodes = probability_vec[:node]
            prob_edges = realized_adj_matrix[node,:node]
            # print(prob_edges)
            # probability_vec[node] = (1 - torch.prod(1 - (prob_nodes * prob_edges)))
            probability_vec = torch.cat((probability_vec, (1 - torch.prod(1 - (prob_nodes * prob_edges))).reshape((1))),0)

        # print(probability_vec)

        turnout = probability_vec[-1]

        payoffs = []


        for player in range(len(distribution)):
            payoffs.append(distribution[player] * turnout)
            if player > 0:
                payoffs[-1] = (self.num_agents * payoffs[-1]) - torch.sigmoid(effort[player-1])
        payoffs[0] =  self.num_agents * payoffs[0]

        return payoffs


class Producing_Game(Game):
    def __init__(self, params):
        self.num_workers, self.real_hour_cost, self.fake_hour_cost, self.max_hours, self.min_hours = params[0], params[1], params[2], params[3], params[4]


    def play(self, strats):
        for strat in strats[1:]:
            for element in strat:
                if element < self.min_hours:
                    element = self.min_hours
                if element > self.max_hours:
                    element = self.max_hours

        prod_vec = torch.zeros(num_workers)
        hour_vec = torch.zeros(num_workers)
        for worker in range(1, self.num_workers+1):
            hour_vec[worker] = torch.sum(strats[worker])
            prod_vec[worker] = strats[worker][0]
            unprod_vec [worker] = strats[worker][1]


        distribution = params[0][1](hour_vec)
        personal_payoff = params[0][0](torch.sum(prod_vec))
        distribution = (distribution*(torch.sum(prod_vec) - personal_payoff)/(torch.sum(distribution))) - (self.real_hour_cost * prod_vec) - (self.fake_hour_cost * unprod_vec)

        payoffs = [personal_payoff]

        for i in range(distribution.shape[0]):
            payoffs.append(distribution[i])

        return payoffs
