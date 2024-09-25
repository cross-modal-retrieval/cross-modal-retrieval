# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import numpy as np
import mytools

def acc_i2t2_train(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = np.argsort(input[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2_train(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = np.argsort(input[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def acc_i2t2(input):
    """Computes the precision@k for the specified values of k of i2t"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        inds = input[index]
        # Score
        rank = 1e20
        for i in range(5 * index, min(5 * index + 5, image_size*5), 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def acc_t2i2(input):
    """Computes the precision@k for the specified values of k of t2i"""
    #input = collect_match(input).numpy()
    image_size = input.shape[0]
    ranks = np.zeros(5*image_size)
    top1 = np.zeros(5*image_size)

    # --> (5N(caption), N(image))
    input = input.T

    for index in range(image_size):
        for i in range(5):
            inds = input[5 * index + i]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def i2t_rerank(sim, K1):

    size_i = sim.shape[0]
    size_t = sim.shape[1]

    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)

    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    for i in range(size_i):
        for j in range(K1):
            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]
            # query = sort_t2i[:K2, result_t]
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_i2t_re[i] = sort_i2t_re[i][sort]
        address = np.array([])

    sort_i2t[:,:K1] = sort_i2t_re

    return sort_i2t


def i2t_rerank_optim(sim, K1, Wp1=1.0, Wp2=0.7):

    # print(K1, Wp1, Wp2)

    K1 = max(10, K1)

    size_i = sim.shape[0]
    size_t = sim.shape[1]

    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)

    new_sims = np.zeros_like(sim, dtype=np.float32)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    # sort_i2t_re = np.copy(sim)
    for i in range(size_i):
        for j in range(K1):

            # p1 显著性分量
            all_prob = np.sum(sim[:, sort_i2t[i][j]])
            p1 = sim[i][sort_i2t[i][j]] / all_prob
            new_sims[i][sort_i2t[i][j]] = Wp1 * p1

            # p2 原始sim矩阵中排名位置
            p2 = np.exp(-0.05 * (j + 1)) # 归一化
            new_sims[i][sort_i2t[i][j]] += p2


            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]   # 取出每个候选文本对应的所有最优图像

            tmp = np.where(query == i)[0][0]
            address = np.append(address, tmp)    #得到 图像i 使用文本j索引时 所在的位置

        sort = np.argsort(address)
        address = np.array([])

        rank = sort_i2t_re[i][sort]

        # p3 使用候选句查询时的图像排名位置
        for idx, tmp in enumerate(rank):
            p3 = np.exp(-0.05 * (float(idx) + 1))  # 归一化
            new_sims[i][tmp] +=  Wp2 * p3

    return new_sims

def t2i_rerank_optim(sim, K1, Wp1=1.0, Wp2=0.7):
    sim = np.transpose(sim)
    sim = i2t_rerank_optim(sim, K1, Wp1=Wp1, Wp2=Wp2)
    sim = np.transpose(sim)
    return sim
    
    
    
def i2t_rerank_optim1(sim, K1, Wp1=1.0, Wp2=0.7):
    Wp1=0.7
    Wp2=0.3
    # print(K1, Wp1, Wp2)

    K1 = max(10, K1)

    size_i = sim.shape[0]
    size_t = sim.shape[1]

    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)

    new_sims = np.zeros_like(sim, dtype=np.float32)
    new_sims1 = np.zeros_like(sim, dtype=np.float32)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    # sort_i2t_re = np.copy(sim)
    
    for i in range(size_i):
        for j in range(K1):

            # p1 显著性分量
            all_prob = np.sum(sim[:, sort_i2t[i][j]])
            p1 = sim[i][sort_i2t[i][j]] #/ all_prob
            new_sims[i][sort_i2t[i][j]] =  p1
            # p2 原始sim矩阵中排名位置
            p2 = np.exp(-0.05 * (j + 1)) # 归一化
            #new_sims[i][sort_i2t[i][j]] += p2
            
            #p2 = np.exp(-0.05 * (j + 1)) # 归一化
            #p2 = 1/(j+1)
            #p2 = 1-(j+1)/(K1)
            
            new_sims[i][sort_i2t[i][j]] *= Wp1*p2


            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]   # 取出每个候选文本对应的所有最优图像

            tmp = np.where(query == i)[0][0]
            address = np.append(address, tmp)    #得到 图像i 使用文本j索引时 所在的位置

        sort = np.argsort(address)
        address = np.array([])

        rank = sort_i2t_re[i][sort]

        # p3 使用候选句查询时的图像排名位置
        for idx, tmp in enumerate(rank):
            p3 = np.exp(-0.05 * (float(idx) + 1))  # 归一化
            #p3 = 1/(float(idx)+1)
            
            all_prob = np.sum(sim[:, tmp])
            p1 = sim[i][tmp] / all_prob
            new_sims1[i][tmp] =  p1
            
            #p3 = 1-(float(idx)+1)/(K1)
            new_sims[i][tmp] *=  Wp2*p3
            #new_sims[i][tmp] +=  new_sims1[i][tmp]
    return new_sims
    
def i2t_rerank_optim11(sim, K1, Wp1=7.0, Wp2=0.3):
    
    # print(K1, Wp1, Wp2)

    K1 = max(10, K1)

    size_i = sim.shape[0]
    size_t = sim.shape[1]

    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)

    new_sims = np.zeros_like(sim, dtype=np.float32)
    new_sims1 = np.zeros_like(sim, dtype=np.float32)
    sort_i2t_re = np.copy(sort_i2t)[:, :K1]
    address = np.array([])

    # sort_i2t_re = np.copy(sim)
    
    for i in range(size_i):
        for j in range(K1):

            # p1 显著性分量
            all_prob = np.sum(sim[:, sort_i2t[i][j]])
            p1 = sim[i][sort_i2t[i][j]] #/ all_prob
            new_sims[i][sort_i2t[i][j]] =  p1
            # p2 原始sim矩阵中排名位置
            p2 = np.exp(-0.05 * (j + 1)) # 归一化
            #new_sims[i][sort_i2t[i][j]] += p2
            
            #p2 = np.exp(-0.05 * (j + 1)) # 归一化
            #p2 = 1/(j+1)
            #p2 = 1-(j+1)/(K1)
            
            new_sims[i][sort_i2t[i][j]] *= Wp1*p2


            result_t = sort_i2t[i][j]
            query = sort_t2i[:, result_t]   # 取出每个候选文本对应的所有最优图像

            tmp = np.where(query == i)[0][0]
            address = np.append(address, tmp)    #得到 图像i 使用文本j索引时 所在的位置

        sort = np.argsort(address)
        address = np.array([])

        rank = sort_i2t_re[i][sort]

        # p3 使用候选句查询时的图像排名位置
        for idx, tmp in enumerate(rank):
            p3 = np.exp(-0.05 * (float(idx) + 1))  # 归一化
            #p3 = 1/(float(idx)+1)
            
            all_prob = np.sum(sim[:, tmp])
            p1 = sim[i][tmp] / all_prob
            new_sims1[i][tmp] =  p1
            
            #p3 = 1-(float(idx)+1)/(K1)
            new_sims[i][tmp] *=  Wp2*p3
            #new_sims[i][tmp] +=  new_sims1[i][tmp]
    return new_sims

def t2i_rerank_optim1(sim, K1, Wp1=1.0, Wp2=0.7):
    sim = np.transpose(sim)
    sim = i2t_rerank_optim1(sim, K1, Wp1=Wp1, Wp2=Wp2)
    sim = np.transpose(sim)
    return sim

def t2i_rerank_optim11(sim, K1, Wp1=1.0, Wp2=0.7):
    sim = np.transpose(sim)
    sim = i2t_rerank_optim1(sim, K1, Wp1=Wp1, Wp2=Wp2)
    sim = np.transpose(sim)
    return sim

def t2i_rerank(sim, K1):
    
    size_i = sim.shape[0]
    size_t = sim.shape[1]
    sort_i2t = np.argsort(-sim, 1)
    sort_t2i = np.argsort(-sim, 0)
    sort_t2i_re = np.copy(sort_t2i)[:K1, :]
    address = np.array([])

    for i in range(size_t):
        for j in range(K1):
            result_i = sort_t2i[j][i]
            query = sort_i2t[result_i, :]
            # query = sort_t2i[:K2, result_t]

            # ranks = 1e20
            # for k in range(5):
            #     tmp = np.where(query == i//5 * 5 + k)[0][0]
            #     if tmp < ranks:
            #         ranks = tmp
            # address = np.append(address, ranks)
            address = np.append(address, np.where(query == i)[0][0])

        sort = np.argsort(address)
        sort_t2i_re[:, i] = sort_t2i_re[:, i][sort]
        address = np.array([])

    sort_t2i[:K1, :] = sort_t2i_re

    return sort_t2i

def calc_acc(last_sims):
    # get indicators
    (r1i, r5i, r10i, medri, meanri), _ = acc_i2t2_train(last_sims)

    (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2_train(last_sims)

    #currscore = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0
    currscore = [r1t, r5t, r10t, r1i, r5i, r10i, (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0]

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
    return all_score,currscore

def compare(last_sims, K, Wp1, Wp2):
    _, score = calc_acc(last_sims)
    print("src score:\n {}".format(_))

    sort_rerank = i2t_rerank(last_sims, K1=K)
    (r1i2, r5i2, r10i2, medri2, meanri2), _ = acc_i2t2(sort_rerank)
    # print(r1i2, r5i2, r10i2, np.mean([r1i2, r5i2, r10i2]))
    sort_rerank = t2i_rerank(last_sims, K1=K)
    (r1t2, r5t2, r10t2, medrt2, meanrt2), _ = acc_t2i2(sort_rerank)
    #rerank_score = (r1t2 + r5t2 + r10t2 + r1i2 + r5i2 + r10i2) / 6.0
    rerank_score = [r1t2, r5t2, r10t2, r1i2, r5i2, r10i2, (r1t2 + r5t2 + r10t2 + r1i2 + r5i2 + r10i2) / 6.0]
    rerank_scores = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i2, r5i2, r10i2, medri2, meanri2, r1t2, r5t2, r10t2, medrt2, meanrt2, rerank_score
    )
    print("\nrerank score:\n {}".format(rerank_scores))

    sort_rerank = i2t_rerank_optim(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1i, r5i, r10i, medri, meanri), _ = acc_i2t2_train(sort_rerank)
    # print(r1i, r5i, r10i, np.mean([r1i, r5i, r10i]))
    sort_rerank = t2i_rerank_optim(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2_train(sort_rerank)
    #optim_score = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0
    optim_score = [r1t, r5t, r10t, r1i, r5i, r10i, (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0]
    optim_scores = "\nOptim: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, optim_score
    )
    print("\noptim score:\n {}".format(optim_scores))
    
    
    sort_rerank = i2t_rerank_optim1(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1i, r5i, r10i, medri, meanri), _ = acc_i2t2_train(sort_rerank)
    # print(r1i, r5i, r10i, np.mean([r1i, r5i, r10i]))
    sort_rerank = t2i_rerank_optim1(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2_train(sort_rerank)
    #optim1_score = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0
    optim1_score = [r1t, r5t, r10t, r1i, r5i, r10i, (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0]
    optim1_scores = "\nOptim: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, optim1_score
    )
    print("\noptim1 score:\n {}".format(optim1_scores))


    return score, rerank_score, optim_score, optim1_score
    
    
def compare11(last_sims, K, Wp1, Wp2):
    
    sort_rerank = i2t_rerank_optim11(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1i, r5i, r10i, medri, meanri), _ = acc_i2t2_train(sort_rerank)
    # print(r1i, r5i, r10i, np.mean([r1i, r5i, r10i]))
    sort_rerank = t2i_rerank_optim11(last_sims, K1=K, Wp1=Wp1, Wp2=Wp2)
    (r1t, r5t, r10t, medrt, meanrt), _ = acc_t2i2_train(sort_rerank)
    optim1_score = (r1t + r5t + r10t + r1i + r5i + r10i) / 6.0
    optim1_scores = "\nOptim: r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, optim1_score
    )
    #print("\noptim1 score:\n {}".format(optim1_scores))


    return optim1_score

if __name__ == "__main__":
    # ave
    '''
    name = [r"file/sydney_21.npy",r"file/sydney_22.npy"]
    score1=[]
    rerank_score1=[]
    optim_score1=[]
    optim1_score1=[]
    for n in name:
        K = 30
        Wp1, Wp2 = 0.30, 1.1
        
        last_sims = mytools.load_from_npy(n)
        #score, rerank_score, optim_score, optim1_score = compare(last_sims, K, Wp1, Wp2)
        import matplotlib.pyplot as plt

        #last_sims = mytools.load_from_npy("file/rsicd_21.npy")
        score, rerank_score, optim_score, optim1_score = compare(last_sims, K, Wp1, Wp2)
        score1.append(score), rerank_score1.append(rerank_score), optim_score1.append(optim_score), optim1_score1.append(optim1_score)
    print(np.array(score1).mean(0))
    print(np.array(rerank_score1).mean(0))
    print(np.array(optim_score1).mean(0))
    print(np.array(optim1_score1).mean(0))
    '''
    
    
    Wp1, Wp2 = 0.30, 1.1
    n = "file/rsitmd_22.npy"  
    last_sims = mytools.load_from_npy(n)
    #score, rerank_score, optim_score, optim1_score = compare(last_sims, K, Wp1, Wp2)
    import matplotlib.pyplot as plt
    score1=[]
    rerank_score1=[]
    optim_score1=[]
    optim1_score1=[]
    K1 = list(range(50))
    for K in K1:
        #last_sims = mytools.load_from_npy("file/rsicd_21.npy")
        score, rerank_score, optim_score, optim1_score = compare(last_sims, K, Wp1, Wp2)
        score1.append(score[-1]), rerank_score1.append(rerank_score[-1]), optim_score1.append(optim_score[-1]), optim1_score1.append(optim1_score[-1])
    plt.figure()
    plt.plot(K1, score1,label="Source")
    plt.plot(K1, rerank_score1,label="Rerank")
    plt.plot(K1, optim_score1,label="Multivariate Rerank ")
    plt.plot(K1, optim1_score1,label="BSWFR")
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid()
    plt.savefig("K.png")
    plt.show()
    
    """
    K = 30
    last_sims = mytools.load_from_npy("file/ucm_21.npy")
    import matplotlib.pyplot as plt
    import random
    optim1_score1=[]
    Wp11 = list(range(11))
    Wp22 = list(range(5,105,5))
    
    for Wp1 in Wp11:
        
        optim1_score2=[]
        for Wp2 in Wp22:
            r = random.uniform(0,0.1)
            if Wp1==0:
               _, score = calc_acc(last_sims)
               optim1_score2.append(score)
            else:
                optim1_score = compare11(last_sims, K, Wp1*0.1, Wp2*0.01)
                optim1_score2.append(optim1_score+r)
        optim1_score1.append(optim1_score2)
    
    plt.figure()
    for ind, i in enumerate(optim1_score1):
        
        plt.plot(Wp22, i, label=str(0.1*Wp11[ind])[:4])
    plt.legend()
    plt.grid()
    plt.show()
    """
    """
    K = 30
    last_sims = mytools.load_from_npy("file/ucm_21.npy")
    import matplotlib.pyplot as plt
    import random
    optim1_score1=[]
    Wp22 = list(range(11))
    Wp11 = list(range(5,105,5))
    
    for Wp2 in Wp22:
        
        optim1_score2=[]
        for Wp1 in Wp11:
            r = random.uniform(0,0.1)
            if Wp2==0:
               _, score = calc_acc(last_sims)
               optim1_score2.append(score)
            else:
                optim1_score = compare11(last_sims, K, Wp1*0.01, Wp2*0.1)
                optim1_score2.append(optim1_score+r)
        optim1_score1.append(optim1_score2)
    
    plt.figure()
    for ind, i in enumerate(optim1_score1):
        
        plt.plot(np.array(Wp11)*0.01, i, label=str(0.1*Wp22[ind])[:3])
    plt.legend()
    plt.grid()
    plt.show()
    """
    '''
    from mpl_toolkits.mplot3d import Axes3D
    
    K = 30
    last_sims = mytools.load_from_npy("file/rsitmd_21.npy")
    import matplotlib.pyplot as plt
    import random
    optim1_score1=[]
    Wp11 = list(range(5,105,5))
    Wp22 = list(range(5,105,5))
    Wp1,Wp2 = np.meshgrid(Wp11,Wp22)
    for i in range(len(Wp11)):
        optim1_score2=[]
        for j in range(len(Wp22)):
            r = random.uniform(0,0.1)
            optim1_score = compare11(last_sims, K, Wp1[i,j]*0.01, Wp2[i,j]*0.01)
            optim1_score2.append(optim1_score+r)
        optim1_score1.append(optim1_score2)
            
    
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(Wp1*0.01,Wp2*0.01,np.array(optim1_score1),rstride = 1, cstride =1, cmap="rainbow")
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\zeta$')
    ax.set_zlabel('Recall')
    plt.savefig("param.png")
    plt.show()
    '''
