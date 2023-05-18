import torch.nn as nn
import torch
import yaml

class Sub_network(nn.Module):
    def __init__(self,input_ch,output_ch,kernel=3,padding=1,stride=2):
        super(Sub_network, self).__init__()

        self.conv = nn.Conv2d(input_ch,output_ch,kernel,padding=padding,stride=stride)
        self.bn = nn.BatchNorm2d(output_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class Expert_network(nn.Module):
    def __init__(self, config_file):
        super(Expert_network, self).__init__()
        
        self.config_file = config_file
        with open(self.config_file) as cf_file:
            config = yaml.safe_load( cf_file.read())
            
        TOTAL_CLASS = config['stage_1']['num_words'] + config['stage_1']['num_objects'] + config['stage_1']['num_faces']

        self.config_file = config_file
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(64,8,kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)


        self.gate1_layer = nn.Linear(56*56*8,3)
        self.gate2_layer = nn.Linear(28*28*8,3)

        self.experts_1 = nn.ModuleList([nn.Sequential(Sub_network(8,8,stride=1),Sub_network(8,8)) for i in range(3)])
        self.experts_2 = nn.ModuleList([nn.Sequential(Sub_network(8,8,stride=1),Sub_network(8,8)) for i in range(3)])

        self.fc = nn.ModuleList([nn.Linear(14*14*8,TOTAL_CLASS) for i in range(3)])

        self.softmax = nn.Softmax(dim=1)

        if config['stage_1']['use_resnet_weights']:
            model_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.conv1.weight = model_resnet.conv1.weight

    def forward(self, x, features=False,gate1_manual=None, gate2_manual=None,temperature = 1.0):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))

        #STEPS
        #1. Get output of all 3 experts
        #2. Getting the 3 gate values and performing softmax
        #3. Multiplying gate values with respective expert output values
        #4. Summing up multiplied values

        expert_out_1 = [m(x) for m in self.experts_1]

        #print("expert_out_1:",expert_out_1[0].shape)

        if gate1_manual:
            gate1 = torch.zeros(x.size()[0],3)
            gate1[:,gate1_manual-1]=1
        else:
            gate1 = self.softmax(self.gate1_layer(torch.flatten(x,start_dim=1))/temperature)
        out_1 = [gate1[:,i].reshape(-1,1,1,1)*expert_out_1[i] for i in range(gate1.shape[1])]
        out1 = torch.stack(out_1, dim=0).sum(dim=0)

        #print("out1:", out1.shape)

        expert_out_2 = [m(out1) for m in self.experts_2]
        if gate2_manual:
            gate2 = torch.zeros(x.size()[0],3)
            gate2[:,gate2_manual-1]=1
        else:
            gate2 = self.softmax(self.gate2_layer(torch.flatten(out1,start_dim=1))/temperature)
        out_2 = [gate2[:,i].reshape(-1,1,1,1)*expert_out_2[i] for i in range(gate2.shape[1])]
        #out2 = torch.cat(out_2, dim=1)

        #print("expert_out_2:",expert_out_2[0].shape)
        out2 = [self.fc[i](torch.flatten(out_2[i],start_dim=1)) for i in range(3)]
        #print("out2:", out2[0].shape)

        out = torch.stack(out2, dim=0).sum(dim=0)
        #print("out:", out.shape)

        if features:
            return gate1,gate2,out,out_2
        #print("gate1",gate1)
        #print("gate2",gate2)

        return gate1,gate2,out
