import time
import random
import numpy as np
from copy import deepcopy

# PYTORCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

from example.utils import memory
from constants import *

#Data augmentation
from PIL import Image
import torchvision.transforms as transforms
from augs.autoaug import SemNavPolicy

augmentations = [
    transforms.ToPILImage(),
    SemNavPolicy(),
    transforms.ToTensor(),
]

transform = transforms.Compose(augmentations)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()

        self.relu = nn.ReLU()

        self.padding1 = nn.ZeroPad2d((1,1,1,1))
        self.padding2 = nn.ZeroPad2d((1,1,1,1))

        self.conv1a = nn.Conv2d(input_dim, 16, 3)
        self.conv1b = nn.Conv2d(16, 32, 3)
        self.conv2a = nn.Conv2d(32, 64, 3)
        self.conv2b = nn.Conv2d(64, 64, 3)

        self.fc1 = nn.Linear(21*21*64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, output_dim)
        
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv1a(self.padding1(x))
        x = self.relu(x)
        x = self.conv1b(self.padding1(x))
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.pool(x)

        x = self.conv2a(self.padding2(x))
        x = self.relu(x)
        #x = self.conv2b(self.padding2(x))
        #x = self.relu(x)
        #x = self.dropout(x)
        x = self.pool(x)

        x = x.view(-1, 21*21*64)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.dropout(x)

        return self.out(x)

def get_resnet18(output_dim):
    '''
    Accepts rgb only (not rgbd)
    Input Size: (3, 224, 224)
    Output Size: # of actions
    '''

    model = timm.create_model('resnet18', pretrained=True)

    #Freeze other layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace last fc layer (trainable)
    fc_in = model.fc.in_features
    model.fc = nn.Linear(fc_in, output_dim)

    return model

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, outputs, memorySize, discountFactor, learningRate, img_rows, img_cols, img_channels ,useTargetNetwork):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0

        self.initNetworks()

    def initNetworks(self):

        self.model = self.createModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)

        if self.useTargetNetwork:
            self.targetModel = self.createModel()

    def createModel(self):
        
        net = Net(self.img_channels, self.output_size)
        #net = get_resnet18(self.output_size)
        net.cuda()
        net.eval() # default eval to remove dropout

        return net

    def backupNetwork(self, model, backup):

        model_state = deepcopy(model.state_dict())
        backup.load_state_dict(model_state)

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)
        print('update target network')

    # predict Q values for all the actions. 
    def getQValues(self, state):

        with torch.no_grad():

            state = torch.FloatTensor(state)
            state = state.cuda()

            if self.useTargetNetwork:
                predicted = self.targetModel(state)
            else:
                predicted = self.model(state)
            
        return predicted[0].cpu().numpy()

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)


    def getMemorySize(self):
        return self.memory.getCurrentSize()


    def learnOnMiniBatch(self, miniBatchSize):
        #t0 = time.time()
        self.count_steps += 1

        state_batch,action_batch,reward_batch,newState_batch,isFinal_batch\
        = self.memory.getMiniBatch(miniBatchSize)



        # Cast to GPU
        state_batch, newState_batch = map(torch.FloatTensor, [state_batch, newState_batch])
        #for i in range(state_batch.shape[0]):
        #    state_batch[i] = transform(state_batch[i])
        state_batch, newState_batch = map(lambda x: x.cuda(), [state_batch, newState_batch])

        #print(state_batch.size())

        with torch.no_grad():

            qValues_batch = self.model(state_batch)
            isFinal_batch = np.array(isFinal_batch) + 0

            """
            target = reward(s,a) + gamma * max(Q(s')
            """
            if self.useTargetNetwork:
                qValuesNewState_batch = self.targetModel(newState_batch)
            else :
                qValuesNewState_batch = self.model(newState_batch)
            qValuesNewState_batch = qValuesNewState_batch.cpu().numpy()

        #print('qvalues_new_state_size: ', qValuesNewState_batch.shape)

        Y_sample_batch = reward_batch + (1 - isFinal_batch) * self.discountFactor * np.max(qValuesNewState_batch, axis=1)
        X_batch = state_batch
        Y_batch = qValues_batch

        for i,action in enumerate(action_batch):
            Y_batch[i][action] = Y_sample_batch[i]

        # Update model parameters
        self.model.train()

        optimizer = self.optimizer
        optimizer.zero_grad()

        preds = self.model(X_batch)

        mse_loss = nn.MSELoss()
        loss = mse_loss(preds, Y_batch)

        loss.backward()
        optimizer.step()

        self.model.eval()

        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()

    def saveModel(self, path):
        if self.useTargetNetwork:
            torch.save(self.targetModel.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def loadWeights(self, path):

        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        if self.useTargetNetwork:
            self.targetModel.load_state_dict(torch.load(path))
            self.targetModel.eval()


    def feedforward(self,observation,explorationRate):
        qValues = self.getQValues(observation)
        action = self.selectAction(qValues, explorationRate)
        return action






