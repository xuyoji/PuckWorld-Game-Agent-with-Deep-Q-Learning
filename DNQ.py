#write by Allen Xu 2020.6

import numpy as np
import random
import turtle
import time
import math
import queue
import torch
from torch import nn

class Circle:
    def __init__(self, r, center, color):
        self.r = r
        self.c = center
        self.color = color
        self.xbound, self.ybound = 300, 300
        self.counter = 0

    def move(self, d):
        self.c[0] += d[0]
        self.c[1] += d[1]
        if self.c[0] > self.xbound:
            self.c[0] = self.xbound
        elif self.c[0] < 0:
            self.c[0] = 0
        if self.c[1] > self.ybound:
            self.c[1] = self.ybound
        elif self.c[1] < 0:
            self.c[1] = 0       

    def follow(self, target):
        t = target.c
        _ = (t[0]-self.c[0], t[1]-self.c[1])
        s = (_[0]**2 + _[1]**2)**0.5
        factor = math.log(s+1, 16)/s
        d = (factor*_[0], factor*_[1])
        self.move(d)

    def draw(self):
        turtle.pu()
        turtle.color(self.color)
        turtle.goto(self.c[0], self.c[1]-self.r)
        turtle.pd()
        turtle.begin_fill()
        turtle.circle(self.r, None, self.r*2)
        turtle.end_fill()
        turtle.pu()
    
    def random_appear(self):
        if self.counter == 32:
            self.c[0] = random.randint(0, self.xbound)
            self.c[1] = random.randint(0, self.ybound)
            self.counter = 0
        else:
            self.counter += 1

class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.Tanh())
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.Tanh())
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.Tanh())
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.Tanh())
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, out_dim),
            nn.Tanh())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class Agent(Circle):
    def __init__(self, r, center, color, target, barrier, epsilon, alpha, gamma, train_flag=True, load_model=True, player_mode=False):
        super().__init__(r, center, color)
        self.v = [0, 0]
        self.target = target
        self.barrier = barrier
        self.experience_queque = []
        self.state = None
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.model = neuralNetwork(8, 32, 128, 32, 8, 4)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.round = 0
        self.avg_r = 0
        self.rwd = 0
        self.r_list = []
        self.counter =0
        self.train_flag = train_flag
        self.model_flag = load_model
        self.seq = '→↑←↓'
        self.player_mode = player_mode
        if player_mode:
            turtle.onkeypress(lambda: self.adjust_v(1), "w")
            turtle.onkeypress(lambda: self.adjust_v(3), "s")
            turtle.onkeypress(lambda: self.adjust_v(2), "a")
            turtle.onkeypress(lambda: self.adjust_v(0), "d")
            turtle.listen()

    def draw(self, r):
        ratio = (r*15 + 1) /2
        self.color = (0.2+0.8*(1-ratio), 0.2+0.8*ratio, 0.4)
        super().draw()
        turtle.pd()
        turtle.color(0, 0, 0)
        turtle.pu()
        turtle.goto(self.c[0]+1, self.c[1]-4)
        turtle.pd()
        turtle.write(self.seq[self.a], False, align="center", font=("Helvetica", 18, "bold"))
        turtle.pu()
        turtle.goto(5,3)
        turtle.color(0.1, 0.1, 0.1)
        ratio = float(self.rwd*15)
        avg_ration = float(self.avg_r*15)
        turtle.write("r = {:.3}\navg_r = {:.3}".format(ratio, avg_ration), True, align="left", font=("Arial", 14, "normal"))

    def get_reward(self):
        t = self.target.c
        b = self.barrier.c
        d1 = ((t[0]-self.c[0])**2 + (t[1]-self.c[1])**2)**0.5
        d2 = ((b[0]-self.c[0])**2 + (b[1]-self.c[1])**2)**0.5
        r1 = -d1/450 + 1+ 1/(d1+1)**0.5
        r2 = (d2<self.barrier.r) * (d2/(self.barrier.r)-2)
        return (r1+r2)/30
    
    def load_model(self):
        self.model.load_state_dict(torch.load('trained_network.pth'))

    def open_log(self):
        self.file = open('./reward.log', 'w')
        self.file.write('r'+'avg_r'+'\n')

    def adjust_v(self, direction):
        if direction==0: #right
            self.v[0] += 2
        elif direction==1: #up
            self.v[1] += 2
        elif direction==2: #left
            self.v[0] -= 2
        elif direction==3: #down
            self.v[1] -= 2
        s=self.c
        if s[0]==0 and self.v[0] < 0:
            self.v[0] = 0
        elif s[0]==self.xbound and self.v[0]>0:
            self.v[0] = 0
        if s[1]==0 and self.v[1]<0:
            self.v[1] = 0
        elif  s[1]==self.ybound and self.v[1]>0:
            self.v[1] = 0
        self.v[0] *= 0.9
        self.v[1] *= 0.9
    
    def get_best_action(self, s):
        #return q, a
        self.model.eval()
        out = self.model(torch.FloatTensor(s))
        q = float('-inf')
        a = -1
        for i, v in enumerate(out):
            if v > q:
                if s[0]==0 and i==2:
                    continue
                elif s[0]==self.xbound and i==0:
                    continue
                elif s[1]==0 and i==3:
                    continue
                elif  s[1]==self.ybound and i==1:
                    continue
                q, a = v, i
        #print(out)      
        if q==0:
            a = random.randint(0, 3)
        self.model.train()
        #elif(q != 0):
            #print(q, a)
        return q, a
                
    def generate_action(self, s):
        r = random.random()
        if r < self.epsilon/(1+math.log(self.round, 200)): 
            a = random.randint(0, 3)
        else:
            q, a = self.get_best_action(s)
        return a

    def my_loss(self, out, target):
        a = target-out
        if a > 1:
            a = torch.FloatTensor(1)
        elif a < -1:
            a = torch.FloatTensor(-1)
        loss = a**2
        loss.requires_grad_()
        return loss

    def train_network(self):
        self.model.train()
        samples = random.sample(self.experience_queque, 20)
        #running_loss = 0.0
        for i, exp in enumerate(samples):
            s, a, r, nxt_s = exp
            # forwarding
            nxt_q, nxt_a = self.get_best_action(nxt_s)
            out = self.model(torch.FloatTensor(s))
            q = out[a]
            loss = self.my_loss(q, r+self.gamma*nxt_q)
            #print(loss)
            #running_loss += loss.item()
            # backwarding
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()    

    def refresh_state(self):
        self.round += 1
        self.state  = tuple(self.c + [30*_ for _ in self.v] + self.target.c + self.barrier.c)
        self.a = self.generate_action(self.state)
        if (self.round % 32==0):
            self.file.write(str(self.avg_r)+'\n')
            #print(self.avg_r)
        if (self.train_flag and self.round % 5000 ==0):
            torch.save(self.model.state_dict(), './network'+str(self.round)+'.pth')

    def step(self):
        r = self.get_reward()
        #print(a)
        if not self.player_mode:
            self.adjust_v(self.a)
        self.move(self.v)
        self.barrier.follow(self)
        turtle.clear()
        self.barrier.draw()
        self.target.draw()
        self.draw(r)
        self.r_list.append(r)

    def cal_r(self):
        self.rwd = sum(self.r_list)/len(self.r_list)
        self.avg_r += (self.rwd-self.avg_r)/self.round
        self.r_list = []

    def train(self):
        nxt_state  = tuple(self.c + [30*_ for _ in self.v] + self.target.c + self.barrier.c)
        transition = (self.state, self.a, self.rwd, nxt_state)
        self.experience_queque.insert(0, transition)
        if len(self.experience_queque)>500000:
            self.experience_queque.pop()
        if len(self.experience_queque)>100:
            self.train_network()

    def run(self):
        print('input \'stop\' to end the procedure and save model and log')
        assert(self.model_flag or self.train_flag)
        self.open_log()
        if self.model_flag:
            self.load_model()
            self.epsilon = 0
        self.running = True
        def loop():
            if self.running:
                if self.counter==0:
                    self.refresh_state()
                    self.counter += 1
                elif self.counter==9:
                    self.cal_r()
                    if self.train_flag:
                        self.train()
                    self.target.random_appear()
                    self.counter=0
                else:
                    self.step()
                    self.counter += 1
                turtle.ontimer(loop, 1)
        loop()
        while True:
            a = input()
            if a == 'stop':
                self.running = False
                break
        self.finish()

    def finish(self):
        self.file.close()
        self.model.eval()
        if self.train_flag:
            torch.save(agent.model.state_dict(), './neural_network.pth')

llx, lly, urx, ury = 0, 0, 300, 300
turtle.setup(width=700, height=700)
turtle.setworldcoordinates(llx, lly, urx, ury)
turtle.mode('world')
turtle.bgcolor(0.9, 0.9, 0.9) 
turtle.ht()
turtle.speed(0)
turtle.tracer(0, 0)

target = Circle(20, [100,200], 'green')
barrier = Circle(80, [0,0], (0.95, 0.8, 0.8))


load_model = True
Train = False
Player_mode = False
alpha = 0.01
initial_epsilon = 0.2
gamma = 0.9
agent = Agent(10, [30, 80], 'blue', target, barrier, initial_epsilon, alpha, gamma, Train, load_model, Player_mode)
agent.run()