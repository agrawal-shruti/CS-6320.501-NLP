# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:53:53 2019

@author: Shruti
"""
import sys

def initialize():
    transition_matrix = {
            'Pie' : {'NNP' : 0.2767, 'MD' : 0.0006, 'VB' : 0.0031, 'JJ' : 0.0453, 'NN' : 0.0449, 'RB' : 0.0510, 'DT' : 0.2026},
            'NNP' : {'NNP' : 0.3777, 'MD' : 0.0110, 'VB' : 0.0009, 'JJ' : 0.0084, 'NN' : 0.0584, 'RB' : 0.0090, 'DT' : 0.0025},
            'MD' : {'NNP' : 0.0008, 'MD' : 0.0002, 'VB' : 0.7968, 'JJ' : 0.0005, 'NN' : 0.0008, 'RB' : 0.1698, 'DT' : 0.0041},
            'VB' : {'NNP' : 0.0322, 'MD' : 0.0005, 'VB' : 0.0050, 'JJ' : 0.0837, 'NN' : 0.0615, 'RB' : 0.0514, 'DT' : 0.2231},
            'JJ' : {'NNP' : 0.0366, 'MD' : 0.0004, 'VB' : 0.0001, 'JJ' : 0.0733, 'NN' : 0.4509, 'RB' : 0.0036, 'DT' : 0.0036},
            'NN' : {'NNP' : 0.0096, 'MD' : 0.0176, 'VB' : 0.0014, 'JJ' : 0.0086, 'NN' : 0.1216, 'RB' : 0.0177, 'DT' : 0.0068},
            'RB' : {'NNP' : 0.0068, 'MD' : 0.0102, 'VB' : 0.1011, 'JJ' : 0.1012, 'NN' : 0.0120, 'RB' : 0.0728, 'DT' : 0.0479},
            'DT' : {'NNP' : 0.1147, 'MD' : 0.0021, 'VB' : 0.0002, 'JJ' : 0.2157, 'NN' : 0.4744, 'RB' : 0.0102, 'DT' : 0.0017}            
            }
    
    observation_matrix = {
            'NNP' : {'Janet' : 0.000032, 'will' : 0, 'back' : 0, 'the' : 0.000048, 'bill' : 0},
            'MD' : {'Janet' : 0, 'will' : 0.308431, 'back' : 0, 'the' : 0, 'bill' : 0},
            'VB' : {'Janet' : 0, 'will' : 0.000028, 'back' : 0.000672, 'the' : 0, 'bill' : 0.000028},
            'JJ' : {'Janet' : 0, 'will' : 0, 'back' : 0.000340, 'the' : 0, 'bill' : 0},
            'NN' : {'Janet' : 0, 'will' : 0.000200, 'back' : 0.000223, 'the' : 0, 'bill' : 0.002337},
            'RB' : {'Janet' : 0, 'will' : 0, 'back' : 0.010446, 'the' : 0, 'bill' : 0},
            'DT' : {'Janet' : 0, 'will' : 0, 'back' : 0, 'the' : 0.506099, 'bill' : 0}
            }
        
    state_space = ['NNP','MD','VB','JJ','NN','RB','DT']
    observation_space = ['Janet','will','back','the','bill']
        
    return transition_matrix, observation_matrix, state_space, observation_space
        
def viterbi(transition_matrix, observation_matrix, state_space, observation_space, obs):
    
    N = len(state_space)
    T = len(obs) 
    obs_num = 0   
    
    viterbi_trellis = {}
    viterbi_trellis[obs[obs_num]] = {}
    
    backtrack = {}
    backtrack[obs[obs_num]] = {}
    print("For observation: " + obs[obs_num])
    for s in state_space:
        #viterbi_trellis[obs[obs_num]] = {}
        viterbi_trellis[obs[obs_num]][s] = transition_matrix['Pie'][s] * observation_matrix[s][obs[obs_num]]
        print("State: " + s + " P(" + s + "|Start): " + str(transition_matrix['Pie'][s]) + " P(" + obs[obs_num] + "|" + s + "): " + str(observation_matrix[s][obs[obs_num]]) + "\tProbability = " + str(viterbi_trellis[obs[obs_num]][s]))
        #print(viterbi_trellis) 
        backtrack[obs[obs_num]][s] = 0
        
    
    for obs_num in range(1,T):
        print()
        print("For observation: " + obs[obs_num])
        viterbi_trellis[obs[obs_num]] = {}
        backtrack[obs[obs_num]] = {}
        
        for state in range(0,N):
            maxtrellis, b = 0, 0
            for prev_state in range(0,N):
                trellis = viterbi_trellis[obs[obs_num-1]][state_space[prev_state]]  * transition_matrix[state_space[prev_state]][state_space[state]] * observation_matrix[state_space[state]][obs[obs_num]]
                if trellis>maxtrellis:
                    maxtrellis = trellis
                    b = prev_state
                    #print("For "+obs[obs_num]+" and state: " + state_space[state] + " updating prev state to: " + state_space[prev_state])
            backtrack[obs[obs_num]][state_space[state]] = b
            print("State: " + state_space[state] + " P(" + state_space[state] + "|" + state_space[b] + "): " + str(transition_matrix[state_space[b]][state_space[state]]) + " P(" + obs[obs_num] + "|" + state_space[state] + "): " + str(observation_matrix[state_space[state]][obs[obs_num]]) + "\tProbability = " + str(maxtrellis))
            viterbi_trellis[obs[obs_num]][state_space[state]] = maxtrellis
    print(backtrack)            
    #termination step
    #this does not have any value here because there is no emission probability given for the final "end" state
    #calculation shown for showing complete algorithm. This step makes no difference here.    
    maxtrellis, b = 0, 0
    for i in range(0,N):
        trellis = viterbi_trellis[obs[T-1]][state_space[i]]
        if maxtrellis<trellis:
            maxtrellis = trellis
            b = i
    print(state_space[b])
    prob_seq = []
    prob_seq.insert(1,state_space[b])
    for i in range (T-1, 0, -1):
        b = backtrack[obs[i]][state_space[b]]
        prob_seq.insert(0,state_space[b])
        #print(i-1, prob_seq)
    #for i in range(0,T):
    print ("Sequence of tags for the observations: ")
    print(list(zip(obs, prob_seq)))
    #print ("Prob ", maxtrellis, "\n")

    
    return 0

if __name__ == '__main__':
    #obs = ["Janet" , "will", "back", "the", "bill"]
    #file = open('D:\\UTD\\SEM 3\\NLP\\HW2_ShrutiAgrawal\\HW3_ShrutiAgrawal\\output.txt', 'a')
    #sys.stdout = file
    while True:
        seq = input ("Enter test sequence or 'exit' to exit: ")
        if seq == "exit":
            break
        else:
            #print(seq)
            obs = []
            obs = [word for word in seq.split(" ")]
            print(obs)
            transition_matrix, observation_matrix, state_space, observation_space = initialize()
            viterbi( transition_matrix, observation_matrix, state_space, observation_space, obs )
            print("----------------------------------------------------------------------------------\n")
    #file.close()
    


