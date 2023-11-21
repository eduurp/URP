import matplotlib.pyplot as plt

opt = []
vec = []

with open('vrf_MIRT.txt', 'r') as file:
    for line in file:
        words = line.strip().split(':')
        opt = int(words[0].strip())
        vec = words[1].strip().strip('[]').split(' ')
        
        v = []
        for v_i in vec:
            if v_i: v.append(float(v_i.strip()))

        plt.scatter(v[0], v[1], color = 'red' if opt==1 else 'blue')

plt.show()