import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")


#Hypothesis of Acquired Immunity
@nb.njit
def run_hypothesis2(trials, surv_prob, rounds=21):
    #Stores the number of resistant bacteria in each trial
    resistant_bacts = np.zeros((trials,))
    for it in range(trials):
        count_res_bacts = 0. #Initial number of resistant bacteria
        #Every round twice the number of bacteria is added
        for bact in range(int(2**rounds)): #Loop over all bacterias
            if np.random.random() < surv_prob: #Probability of survival on attack of virus
                count_res_bacts = count_res_bacts  + 1.
        resistant_bacts[it] = count_res_bacts
    return(resistant_bacts)


#Hypothesis of Immunity by Mutation
@nb.njit
def run_hypothesis1(trials, mut_prob, rounds=21):
    #Stores the number of resistant bacteria in each trial
    resistant_bacts = np.zeros((trials,))
    for it in range(trials):
        bact_count = 1.#Simulation starts with one bacteria
        res_bact_count = 0.#No mutated bacteria in first round
        for rnd in range(rounds):#Loop over 21 rounds
            bact_count = bact_count * 2
            #Proliferation of mutated bacteria
            res_bact_count = res_bact_count * 2
            for bact in range(bact_count):
                #Mutation to resistance
                if np.random.random() < mut_prob:
                    res_bact_count = res_bact_count  + 1.
        resistant_bacts[it] = res_bact_count
        
    return(resistant_bacts)

#--------------------------------------------
# Functions for making movie
#--------------------------------------------

# Movie for hypthesis 2
def update_hypo2(frame):
    plt.clf()
    res_bact2 = run_hypothesis2(10000, a, frame)
    plt.hist(res_bact2, bins=np.arange(np.min(res_bact2)-0.5,
                                       np.max(res_bact2)+0.5, 1),
             label="a="+str(a)+" t="+str(frame))
    plt.xlabel("Number of resistant bacteria")
    plt.ylabel("Number of trials")
    plt.legend()
    plt.title("Hypothesis of Acquired Immunity  Generation = "
              +str(frame))
    print(frame)
    
# Movie for hypothesis 2
def update_hypo1(frame):
    plt.clf()
    res_bact1 = run_hypothesis1(10000, a, frame)
    plt.yscale("log")
    plt.xlabel("Number of resistant bacteria")
    plt.ylabel("Number of trials")
    plt.title("Hypothesis of Mutation to Immunity  Generation = "
              +str(frame))
    diff = np.max(res_bact1)-np.min(res_bact1)
    plt.hist(res_bact1, label="a="+str(a)+" t="+str(frame))
    print(frame)    
#--------------------------------------------
    
if __name__=="__main__":
    import time
    t = time.time()
    
    a = 0.0001 #Mutation rate of an individual bacteria for unit time

    #----------------------------------------
    #Plot probability distribution of hypothesis 1
    plt.clf()
    res_bact = run_hypothesis2(10000, a)
    plt.hist(res_bact, bins=np.arange(np.min(res_bact)-0.5,
                                      np.max(res_bact)+0.5, 1),
             label="a="+str(a)+" t="+str(20))
    plt.xlabel("Number of resistant bacteria")
    plt.ylabel("Number of trials")
    plt.legend()
    plt.title("Hypothesis of Acquired Immunity  Generation = "
              +str(21))
    plt.savefig("Hypo2.png")
    plt.show()

    #Plot probability distribution of hypothesis 2
    plt.clf()
    res_bact2 = run_hypothesis1(10000, a)
    plt.yscale("log")
    plt.xlabel("Number of resistant bacteria")
    plt.ylabel("Number of trials")
    plt.title("Hypothesis of Mutation to Immunity  Generation = "
              +str(21))
    diff = np.max(res_bact2)-np.min(res_bact2)
    plt.hist(res_bact2, label="a="+str(a)+" t="+str(20))
    plt.legend()
    plt.savefig("Hypo1.png")
    plt.show()
    #-----------------------------------------
    print(time.time()-t)


    
    #Movie generation hypothesis 2------------
    t = time.time()
    fig, ax = plt.subplots()
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=2,
                    metadata=dict(artist='Anuran'), bitrate=1800)

    ani = matplotlib.animation.FuncAnimation(fig, update_hypo2,
                                             frames=np.arange(1,22,1),
                                         interval=10, repeat=False)
    ani.save('hypo2.mp4', writer=writer)

    plt.show()

    #Movie generation hypothesis 1------------
    fig, ax = plt.subplots()
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=2,
                    metadata=dict(artist='Anuran'), bitrate=1800)

    ani = matplotlib.animation.FuncAnimation(fig, update_hypo1,
                                             frames=np.arange(1,22,1),
                                         interval=10, repeat=False)
    plt.show()
    ani.save('hypo1.mp4', writer=writer)
    #------------------------------------------
    print(time.time()-t)
    
