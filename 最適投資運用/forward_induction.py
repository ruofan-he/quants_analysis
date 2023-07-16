import numpy as np

if __name__ == '__main__':
    
    c = 0.02; # 取引コスト すなわち 売買手数料では購入や売買額面の(1+c)倍コストがかかるとする。
    M = 100; # 初期資金
    price_process = [[1,1,1,1,1], # Money market account
                     [1,2,1,1,1], # product 1
                     [1,1,1,2,1]] # product 2
    
    price_process = np.array(price_process)
    print(price_process)

    money_process = np.zeros_like(price_process)
    money_process[0,0] = M # 初期にはMoney market accoutにてMの資金を持つとする。
    
    for t in range(1, len(price_process[0])):
        for i in range(len(price_process)): # to
            candidate_list = []
            for j in range(len(price_process)): # from
                money = None
                if i == j:
                    money = price_process[i,t]/price_process[i,t-1]*money_process[j,t-1]
                if i != j:
                    if i == 0 or j == 0:
                        money = price_process[i,t]/price_process[i,t-1]*money_process[j,t-1]/(1+c)
                    else:
                        money = price_process[i,t]/price_process[i,t-1]*money_process[j,t-1]/(1+c)**2
                candidate_list.append(money)
            money_process[i,t] = np.max(candidate_list)

    print(money_process)
