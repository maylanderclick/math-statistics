import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

V = int(input('Variant: '))

N = 200  # количество сгенерированных чисел
n = 5 + V % 16
p = 0.3 + 0.005 * V
l = 0.5 + 0.01 * V
print('p = {}\nn = {}\nl = {}'.format(p, n, l))


# binom_not_sort = numpy.random.binomial(n, p, N)
# binom = numpy.sort(binom_not_sort)
binom = [5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
         8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
         9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
         11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
         12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
         13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
         14, 14, 14, 14, 14, 14, 15, 15]

# geom_not_sort = numpy.random.geometric(p, N)
# geom = numpy.sort(geom_not_sort)
geom = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4]


# poisson_not_sort = numpy.random.poisson(l, N)
# pois = numpy.sort(poisson_not_sort)

pois = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5]

#print('binom', binom)
#print('geom', geom)
#print('pois', pois)

#######
# BIN
#######

s = 0
binom_stat_s = numpy.zeros((4, max(binom)+1), dtype=numpy.object)

print('min binom', min(binom))
# заполняем массив binom_stat_s
for i in range(min(binom), max(binom)+1):
    period = 0
    w = 0
    for k in range(N):
        if binom[k] == i:
            period += 1
        else:
            continue
    w = period/N
    s = s + w
    binom_stat_s[0][i] = int(i)
    binom_stat_s[1][i] = int(period)
    binom_stat_s[2][i] = float("{0:.4f}".format(w))
    binom_stat_s[3][i] = float("{0:.4f}".format(s))

print('binom_state_s:')
for i in range(len(binom_stat_s)):
    for j in range(len(binom_stat_s[i])):
        print(binom_stat_s[i][j], end=' ')
    print()

print('\nbinomial static sequence: ')
binom_x = binom_stat_s[0]
binom_n = binom_stat_s[1]
binom_w = binom_stat_s[2]

fig, axes = plt.subplots(3, 2)

axes[0, 0].plot(binom_x, binom_w)
axes[0, 0].set(xlabel='w', ylabel='x')
axes[0, 0].set(title='Полигон относит. частот')
axes[0, 0].set_xbound(0, max(binom))
print('binom_w', binom_w)
axes[0, 0].set_ybound(0, binom_w[-1])
axes[0, 0].set_xticks(range(1, n))
sort_binom_w = binom_w.copy()
sort_binom_w.sort()
axes[0, 0].set_yticks(
    [(i * int(i % 2 == 0))/100 for i in range(0, int(sort_binom_w[-1]*100)+5)])

axes[0, 1].set(title='Эмпир. функция распределения')
print('binom_w_2', binom_w)
axes[0, 1].hlines(numpy.cumsum(binom_w), range(
    int(binom_x[-1])+1), range(1, int(binom_x[-1])+2))
axes[0, 1].set_xbound(0, max(binom)+1)
axes[0, 1].set_ybound(0, 1.1)
axes[0, 1].set_xticks(range(1, int(binom_x[-1]+1)))
axes[0, 1].set_yticks([(x*(x % 2 == 0))/10 for x in range(11)])

q = 1 - p
print('\nBinomial characteristics:')
print('Мат ожидание: {:.5f}'.format(float(n*p)))
print('Дисперсия: {:.5f}'.format(float(n*p*q)))
print('Среднее квадратическое отклонение: {:.5f}'.format(float(n*p*q)))
print('Мода: {:.5f}'.format(
    float((n+1)*p if int((n+1)*p) != float((n+1)*p) else ((n+1)*p) - 0.5)))
print('Медиана: {:.5f}'.format(round(n*p)))
print('Коэффициент асимметрии:  {:.5f}'.format(float((q-p)/math.sqrt(n*p*q))))
print('Коэффициент эксцесса: {:.5f}'.format(float((1 - 6*p*q)/(n*p*q))))

#######
# GEOM
#######
s = 0
geom_stat_s = numpy.zeros((4, max(geom)+1), dtype=numpy.object)

print('min geom', min(geom))
# заполняем массив geom_stat_s
for i in range(min(geom), max(geom)+1):
    period = 0
    w = 0
    for k in range(N):
        if geom[k] == i:
            period += 1
        else:
            continue
    w = period/N
    s = s + w
    geom_stat_s[0][i] = int(i)
    geom_stat_s[1][i] = int(period)
    geom_stat_s[2][i] = float("{0:.4f}".format(w))
    geom_stat_s[3][i] = float("{0:.4f}".format(s))

print('geom_state_s:')
for i in range(len(geom_stat_s)):
    for j in range(len(geom_stat_s[i])):
        print(geom_stat_s[i][j], end=' ')
    print()

print('\ngeometric static sequence: ')
geom_x = geom_stat_s[0]
geom_n = geom_stat_s[1]
geom_w = geom_stat_s[2]

axes[1, 0].plot(geom_x, geom_w)
axes[1, 0].set(xlabel='w', ylabel='x')
axes[1, 0].set_xbound(0, max(geom))
print('geom_w', geom_w)
axes[1, 0].set_ybound(0, geom_w[-1])
axes[1, 0].set_xticks(range(0, int(max(geom_x)+1)))
sort_geom_w = geom_w.copy()
sort_geom_w.sort()
axes[1, 0].set_yticks(
    [(i * int(i % 2 == 0))/100 for i in range(0, int(sort_geom_w[-1]*100)+11, 9)])

print('numpy.cumsum(geom_w)', numpy.cumsum(geom_w))
axes[1, 1].hlines(numpy.cumsum(geom_w), range(
    int(geom_x[-1])+1), range(1, int(geom_x[-1])+2))
axes[1, 1].set_xbound(0, max(geom_x)+1)
axes[1, 1].set_ybound(0, 1.1)
axes[1, 1].set_xticks(range(0, int(geom_x[-1]+1)))
axes[1, 1].set_yticks([(x*(x % 2 == 0))/10 for x in range(11)])

q = 1 - p
print('\nGeometric characteristics:')
print('Мат ожидание: {:.5f}'.format(float(q/p)))
print('Дисперсия: {:.5f}'.format(float(q/p**2)))
print('Среднее квадратическое отклонение: {:.5f}'.format(
    float(math.sqrt(q)/p)))
print('Мода: 0')
print('Медиана: {:.5f}'.format(-(math.log1p(2)/math.log1p(q))
                               if int(math.log1p(2)/math.log1p(q)) == float(math.log1p(2)/math.log1p(q))
                               else -(math.log1p(2)/math.log1p(q)) - 0.5))
print('Коэффициент асимметрии:  {:.5f}'.format(float((2-p)/math.sqrt(q))))
print('Коэффициент эксцесса: {:.5f}'.format(float((6 + ((p**2)/q)))))

#######
# POIS
#######
s = 0
pois_stat_s = numpy.zeros((4, max(pois)+1), dtype=numpy.object)

print('min pois', min(pois))
# заполняем массив pois_stat_s
for i in range(min(pois), max(pois)+1):
    period = 0
    w = 0
    for k in range(N):
        if pois[k] == i:
            period += 1
        else:
            continue
    w = period/N
    s = s + w
    pois_stat_s[0][i] = int(i)
    pois_stat_s[1][i] = int(period)
    pois_stat_s[2][i] = float("{0:.4f}".format(w))
    pois_stat_s[3][i] = float("{0:.4f}".format(s))

print('pois_state_s:')
for i in range(len(pois_stat_s)):
    for j in range(len(pois_stat_s[i])):
        print(pois_stat_s[i][j], end=' ')
    print()

print('\npoisson static sequence: ')
pois_x = pois_stat_s[0]
pois_n = pois_stat_s[1]
pois_w = pois_stat_s[2]

axes[2, 0].plot(pois_x, pois_w)
axes[2, 0].set(xlabel='w', ylabel='x')
print('pois_w', pois_w)
axes[2, 0].set_ybound(0, pois_w[-1])
axes[2, 0].set_xticks(range(0, int(max(pois_x)+1)))
sort_pois_w = pois_w.copy()
sort_pois_w.sort()
axes[2, 0].set_yticks(
    [(i * int(i % 2 == 0))/100 for i in range(0, int(sort_pois_w[-1]*100)+11, 9)])

print('numpy.cumsum(pois_w)', numpy.cumsum(pois_w))
axes[2, 1].hlines(numpy.cumsum(pois_w), range(
    int(pois_x[-1])+1), range(1, int(pois_x[-1])+2))
axes[2, 1].set_xbound(0, max(pois_x)+1)
axes[2, 1].set_ybound(0, 1.1)
axes[2, 1].set_xticks(range(0, int(pois_x[-1]+1)))
axes[2, 1].set_yticks([(x*(x % 2 == 0))/10 for x in range(11)])

plt.show()

q = 1 - p
print('\nPoisson characteristics:')
print('Мат ожидание: {:.5f}'.format(float(l)))
print('Дисперсия: {:.5f}'.format(float(l)))
print('Среднее квадратическое отклонение: {:.5f}'.format(
    float(math.sqrt(l))))
print('Мода: {:.5f}'.format(float(l)))
print('Медиана: {:.5f}'.format(float(l + 1/3 - 0.002/l)))
print('Коэффициент асимметрии:  {:.5f}'.format(float(1/math.sqrt(l))))
print('Коэффициент эксцесса: {:.5f}'.format(float(1/l)))
