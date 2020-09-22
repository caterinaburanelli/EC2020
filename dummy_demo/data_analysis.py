import statistics
import matplotlib.pyplot as plt
import matplotlib.style as style

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

enemy = "2"
generations = []
for i in range(16):
        generations.append(i)
        values_T = data_analysis(enemy, "Tournement", str(i))
for i in range(16):
        values_A = data_analysis(enemy, "All", str(i))

# style.use('seaborn-poster') #sets the size of the charts
# style.use('ggplot')

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

for i in range (15):
    print(values_A[0], values_A[1])
    if i == 14:
        plt.errorbar(generations[i], values_T[0][i], yerr=values_T[1][i], ecolor="Blue", elinewidth=3, capsize=0, alpha = 0.25, label = "Std_Tour")
        plt.errorbar(generations[i], values_A[0][i], yerr=values_A[1][i], ecolor="Green", elinewidth=3, capsize=0, alpha = 0.25, label = "Std_All")
    else:
        plt.errorbar(generations[i], values_T[0][i], yerr=values_T[1][i], ecolor="Blue", elinewidth=3, capsize=0, alpha = 0.25)
        plt.errorbar(generations[i], values_A[0][i], yerr=values_A[1][i], ecolor="Green", elinewidth=3, capsize=0, alpha = 0.25)
plt.title(f"Results for enemy {enemy}")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.show()