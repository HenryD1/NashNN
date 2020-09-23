#@title PD testing

class Player:
    pass

class PD_Worker(Player):
    def __init__(self):
        self.mat1 = (2*torch.rand([2,1]) - 1)
        self.mat1.requires_grad = True
        self.b = (2*torch.rand([2]) - 1)
        self.b.requires_grad = True
    def get_params(self):
        return [self.mat1,self.b]

    # x represents our data
    def forward(self, x):
      output = (self.mat1 @ x) + self.b

      return output.reshape([-1])


#@title PD testing
'''
class PD_Worker(Player):
    def __init__(self):
      super(PD_Worker, self).__init__()
      self.fc1 = nn.Linear(1, 2)
      self.params =

    # x represents our data
    def forward(self, x):
      output = (self.fc1(x))

      return output.reshape([-1])
'''

class Network_Worker(Player):
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
