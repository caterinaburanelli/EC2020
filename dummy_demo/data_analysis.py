import statistics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from scipy import stats
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes


mean_generation_valuesAll =[]
mean_generation_valuesTournement = []
std_generation_valuesTournement = []
std_generation_valuesAll = []
high_generation_valuesTournement = []
high_generation_valuesAll = []
def data_analysis(enemy, recombination, gen):
    data_file = f"results_enemy{enemy}{recombination}.txt"
    generation_mean_values = []
    generation_std_values = []
    generation_high_values = []

    with open(data_file) as f:
        for line in (f):
            line = line.split(',')
            if line[0] == gen:
                generation_high_values.append(float(line[1]))
                generation_mean_values.append(float(line[2]))
                generation_std_values.append(float(line[3]))

    mean = statistics.mean(generation_mean_values)
    std = statistics.mean( generation_std_values)
    high = statistics.mean(generation_high_values)

    if recombination == "All":
        std_generation_valuesAll.append(std)
        mean_generation_valuesAll.append(mean)
        high_generation_valuesAll.append(high)
        return (mean_generation_valuesAll, std_generation_valuesAll, high_generation_valuesAll)
    else:
        mean_generation_valuesTournement.append(mean)
        std_generation_valuesTournement.append(std)
        high_generation_valuesTournement.append(high)
        return (mean_generation_valuesTournement, std_generation_valuesTournement, high_generation_valuesTournement)

enemy = "3"
generations = []
for i in range(16):
        generations.append(i)
        values_T = data_analysis(enemy, "Tournement", str(i))
for i in range(16):
        values_A = data_analysis(enemy, "All", str(i))

plt.show()
# style.use('seaborn-poster') #sets the size of the charts
# style.use('ggplot')

def line_plot():
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                CB91_Purple, CB91_Violet]   
    plt.rcParams['font.family'] = "serif"    
    plt.plot(values_T[2], color = "darkBlue", alpha = 1, label = "Max_Tour")
    plt.plot(values_A[2], color = "darkGreen", alpha = 1, label = "Max_All")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
    plt.plot(values_T[0], color = "Blue", alpha = 0.5, label = "Mean_Tour")
    plt.plot(values_A[0], color = "Green", alpha = 0.5, label = "Mean_All")

    for i in range (16):
        print(values_A[0], values_A[1])
        plt.errorbar(generations[i], values_T[0][i], yerr=values_T[1][i], ecolor="Blue", elinewidth=3, capsize=0, alpha = 0.25)
        plt.errorbar(generations[i], values_A[0][i], yerr=values_A[1][i], ecolor="Green", elinewidth=3, capsize=0, alpha = 0.25)
    plt.title(f"Results for enemy {enemy}",fontsize=20)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.legend(loc='lower right', fontsize=14)
    plt.ylabel('Fitness', fontsize=16)
    plt.xlabel('Generation', fontsize=16)
    plt.show()

def box_plot():
    data_file = f"dummy_demo_personal_gain.txt"
    count = 0
    fitness = []
    algorithm = []
    indiviual_gain = []
    enemy = []
    with open(data_file) as f:
        next(f)
        for line in (f):
            line = line.split(',')
            algorithm.append(line[5])
            indiviual_gain.append(float(line[3]))
            enemy.append(line[4])
            fitness.append(line[0])

    style.use('ggplot')
    a = algorithm
    b = indiviual_gain
    c = enemy
    d = fitness

        # intialise data of lists.
    data = {'Algorithm':a,
            'personal_gain':b,
            'Enemy':c,
            'Fitness': d}
            
    # Create DataFrame
    df = pd.DataFrame(data)
    
    a = df.loc[(df['Enemy'] == ' 2') & (df['Algorithm'] == ' All')]['Fitness']
    b = df.loc[(df['Enemy'] == ' 2') & (df['Algorithm'] == ' Tournament')]['Fitness']
    l = []
    l1 = []
    for i in a:
        l.append(float(i))
    for i in b:
        l1.append(float(i))
    print(statistics.mean(l))
    print(statistics.mean(l1))
    print(stats.ttest_ind(l,l1))
    sns.boxplot(x="Enemy", y="personal_gain",data=data,  hue="Algorithm")
    plt.ylabel('Individual gain', fontsize=16)
    plt.xlabel('Enemy', fontsize=16)
    plt.title("Individual gain per enemy", fontsize = 20)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    
    plt.legend(fontsize = 14, loc='upper right')
    plt.show()
box_plot()
line_plot()