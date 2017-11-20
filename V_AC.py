#TODO：结果可视化；车辆数大于7时得到局部最优解：maybe局部更新规则调整
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import copy

#使用经纬度计算两点之间的距离
def haversine(lat0, lon0, lat1, lon1):
  R = 6371 # radius earth
  dLat = lat1-lat0
  dLon = lon1-lon0
  a = math.sin(dLat/2)**2 + math.sin(dLon/2)**2 * math.cos(lat0) * math.cos(lat1)
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  return (R*c)

#返回路径总长度
def GetTotalLength(graph,tour):
  length = 0
  for i in range(len(tour)-1):
    length += graph[tour[i]][tour[i+1]] 
  return length

#返回总长度最短的路径
def checkForBestTour(graph, nodes, tours, oldBestTour):
  best = float('inf')#无穷大
  bestT = []

  for t in tours:
    length = GetTotalLength(graph, t)
    if (length < best):
      best = length
      bestT = t
  if not oldBestTour:
    return bestT
  elif (best <= GetTotalLength(graph, oldBestTour)):
    return bestT
  return oldBestTour

#轮盘法选择下一节点
def probfunc(l):
  probtrans=np.array(l)
  cumsumprobtrans=probtrans.cumsum()
  cumsumprobtrans-=np.random.rand()
  i=list(cumsumprobtrans > 0).index(True)
  return i

#局部信息素更新规则
def localUpdatingRule(pheromone, lastNode, currentNode, tau0val): 
  rho = 0.1
  Q=1
  pheromone[lastNode][currentNode] = (1-rho) * pheromone[lastNode][currentNode] + Q/disttable[lastNode][currentNode]

#全局信息素更新规则
def globalUpdatingRule(graph, pheromone, bestTour):
  for r in range(len(graph)):
    for s in range(len(graph)):
      alpha = 0.1
      nval = (1-alpha) * pheromone[r][s]
      if isEdgeOfBestTour(bestTour, r, s):
        #nval += alpha * (1/GetTotalLength(graph, bestTour))
        nval += 1/GetTotalLength(graph, bestTour)
      pheromone[r][s] = nval

#从当前节点出发选择下一节点
def stateTransitionRule(graph, pheromone, currentNode,reachable, depots):
  #如果可达到点全为仓库，随机选一个；否则根据轮盘法选择
  alpha = 1
  beta = 5
  if set(reachable)&set(depots) == set(depots):
    s=reachable[np.random.randint(len(reachable))]
  else:
    prob=np.zeros(len(reachable))
    for k in range(len(reachable)):
      prob[k] = np.power(pheromone[currentNode][reachable[k]],alpha) * np.power(disttable[currentNode][reachable[k]],beta)
    cumsumprobtrans=(prob/sum(prob)).cumsum()
    cumsumprobtrans-=np.random.rand()
    s=reachable[list(cumsumprobtrans > 0).index(True)]

  return s

#为每个蚂蚁选择下一节点
def chooseNext(graph, pheromone, remaining, tours, depots, maxCapacity, cap, Q):
  ant = 0 # current ant
  for a in tours:
    reachable = []
    for i in remaining[ant]:
      if (cap[ant] >= Q[i] and Q[i] != 0):#剩余量够i节点消费
        reachable.append(i)
    if not reachable: # reachable is empty
      for i in remaining[ant]:
        if cap[ant] >= Q[i]:#需要回仓库重新装载
          reachable.append(i)
    if not reachable:
      continue 
    #print('蚂蚁:',ant,'可达点',reachable)
    oldPos = a[len(a)-1]#目前所在节点
    newPos = stateTransitionRule(graph, pheromone, oldPos, reachable, depots)
    #print('蚂蚁:',ant,'oldPos:',oldPos,'newPos:',newPos)
    cap[ant] -= Q[newPos]
    if newPos in depots:
      cap[ant] = maxCapacity
    localUpdatingRule(pheromone, oldPos, newPos, tau0(graph))
    a.append(newPos)
    remaining[ant].remove(newPos)
    #print(remaining[ant])
    ant += 1

#重置，准备进入下一次迭代
def reset(remaining, tours, nodes, ants, maxCapacity, cap):
  del remaining[:]
  for i in range(len(nodes)):
    remaining.append(nodes[:])

  del tours[:]
  for i in range(ants):
    tours.append([])

  del cap[:]
  for i in range(ants):
    cap.append(maxCapacity)

#随机选择每只蚂蚁出发节点
def positionAnts(ants, tours, numNodes, remaining, cap, Q, depots):
  pos =list(range(numNodes))
  for ant in range(ants):
    p = random.choice(pos)
    tours[ant].append(p)
    remaining[ant].remove(p)
    pos.remove(p)
    cap[ant] -= Q[p] # if ant starts on customer node, substract quantity the customer asks for

#增加v个节点到graph中
def addDepots(v, graph):
  l = []
  for g in graph: # create copy of graph
    l.append(g[:])
  l.pop(0) # remove adjacency list 0, the depots together form list 0
  for i in l:
    for j in range(v-1):
      i.insert(0,i[0]) # insert distance to depots 0,1,...
  for i in range(v):
    tmp = [0 for j in range(v)] # distance between depots if 0
    for j in range(1,len(graph)):
      tmp.append(graph[0][j]) # append length from depot to customer
    l.insert(0, tmp)
  return l

#将一次巡游分割为多次
def splitTours(bestTour, depots):
  tour=bestTour[:-1]
  tmp=0
  tourslist=[]
  for i in range(len(tour)):
    if(tour[i] in depots):
      if tmp != i:
        tourslist.append(tour[tmp:i])
      tmp=i+1
  if tmp != len(tour):
    tourslist.append(tour[tmp:])
  for i in range(len(tourslist)):
    for j in range(len(tourslist[i])):
      tourslist[i][j]-=(len(depots)-1)
  return tourslist

'''调整巡游从仓库开始'''
def adjustTours(tours, depots):
  for t in tours:
    while (t[0] not in depots):
      t.append(t.pop(0)) # left shift (append first item at end)
    t.append(t[0]) # append first node at end -> round trip

if __name__ == '__main__':
  #有13个目标点，1个仓库点 originalgraph[0]表示仓库到其他13个目标点的距离
  originalgraph = [[0, 20, 14, 10, 2, 7, 3, 20, 3, 40, 1, 22, 6, 20],[20, 0, 2, 5, 4, 33, 10, 30, 3, 12, 42,
  19, 8, 21],[14, 2, 0, 10, 3, 22, 10, 3, 2, 33, 23, 7, 27, 5], [10, 5, 10, 0, 6, 20, 20, 11, 21, 21,
  73, 6, 14, 20],[2, 4, 3, 6, 0, 1, 2, 40, 12, 18, 17, 25, 30, 7], [7, 33, 22, 20, 1, 0, 40, 5, 3, 2,
  3, 11, 10, 33],[3, 10, 10, 20, 2, 40, 0, 8, 4, 7, 8, 24, 5, 13], [20, 30, 3, 11, 40, 5, 8, 0, 9, 11,
  4, 12, 3, 19],[3, 3, 2, 21, 12, 3, 4, 9, 0, 12, 42, 33, 21, 18], [40, 12, 33, 21, 18, 2, 7, 11, 12,
  0, 6, 3, 17, 4],[1, 42, 23, 73, 17, 3, 8, 4, 42, 6, 0, 6, 26, 8], [22, 19, 7, 6, 25, 11, 24, 12, 33,
  1, 6, 0, 20, 15],[6, 8, 27, 14, 30, 10, 5, 3, 21, 17, 26, 20, 0, 18],[20, 21, 5, 20, 7, 33, 13, 19,
  18, 4, 8, 15, 18, 0]]

  originalQ = [2, 10, 5, 18, 7, 8, 1, 16, 4, 18, 13, 12, 10]#每个节点所需要的货物量
  maxCapacity = 20 # max capacity of vehicles
  v =input('输入可执行任务的无人机数量:')  #number of vehicles。车辆数目小于7时，结果不正确。因为总的货物总量为124，124/(最大承载量20)=7，当车辆数少于7时，不具备执行任务的条件;
  while int(v) < math.ceil(sum(originalQ)/maxCapacity):#向上取整
    v=input('无人机总承载量不具备执行任务能力，请重新输入:')
  v=int(v)

  graph = addDepots(v, originalgraph) # add depots
  numNodes = len(graph) # number of Nodes

  disttable=copy.deepcopy(graph)
  for i in range(numNodes):
    for j in range(numNodes):
      if disttable[i][j]==0:
        disttable[i][j]+=1e10
  disttable=1/(np.array(disttable))
  #print(disttable,graph)
  nodes = list(range(len(graph))) # node list
  depots = list(range(v)) # list of depots 有多少量车就假定有多少个仓库

  # quantity of goods the customer asks for
  Q = []
  for i in range(numNodes):
    if ((i-len(depots)) < 0):
      Q.append(0)
    else:
      Q.append(originalQ[i-len(depots)])
  #print(Q)

  # number of ants
  if (numNodes < 10):
    ants = numNodes
  else:
    ants = 10

#初始化信息素
  pheromone=np.ones((numNodes,numNodes))

  for i in range(10):
    remaining = [ nodes[:] for i in range(ants) ] # remaining nodes
    tours = [ [] for i in range(ants) ] # generated tours
    cap = [ maxCapacity for i in range(ants) ] # remaining goods of ants
    bestTour = [] # best tour so far
    positionAnts(ants, tours, numNodes, remaining, cap, Q, depots) # position ants on nodes
    #print(tours,remaining)
    #print('节点数量为：%d'%(numNodes))

    for count in range(200):#迭代200次
      for i in range(numNodes):
        chooseNext(graph, pheromone, remaining, tours, depots, maxCapacity, cap, Q)#每只蚂蚁遍历所有节点
      #print(tours,remaining)
      adjustTours(tours, depots) # shift nodes so that depot is at beginning/end
      #print(tours)#返回从仓库出发到仓库的巡游
      bestTour = checkForBestTour(graph, nodes, tours, bestTour)
      #print(bestTour)
      #globalUpdatingRule(graph, pheromone, bestTour) #增加besttour所包括边的信息素
      reset(remaining, tours, nodes, ants, maxCapacity, cap)
      positionAnts(ants, tours, numNodes, remaining, cap, Q, depots) 
    #print(bestTour)
    bestTourTotal = GetTotalLength(graph, bestTour)
    tourslist=splitTours(bestTour,depots)
    #print(tourslist)
    print('length of best tour: ', bestTourTotal)
    numTours=len(tourslist)
    print('无人机数为: ', numTours)
    for i in range(numTours):
      print('第%d架分配的任务点为：%s'%(i+1,str(tourslist[i])))