import pandas as pd
import numpy as np
from apriori import extract
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from functools import reduce


def dynamic_cal(key, target):   # calculate PF for each packet
    kl = key
    SKL = 0
    SKN = 0

    for k in kl:
        offset = target.find(k)
        if offset < 0:  # If there is no keyword in target,
            continue

        if len(k) <= 0:  # Noise canceling
            continue

        SKL += len(k)

        """
        if len(rule) > 0:
            rule.append({"len": offset - rule[i - 1]["len"], "type": i})    # dynamic field size
            rule.append({"len": offset + len(k) + 1, "type": k})    # keyword end position
        else:
            rule.append({"len": offset + len(k) + 1, "type": k})    # keyword end position
        """
        SKN += 1

    if SKL == 0 or SKN == 0:
        return 0, 0, 0

    SKL = SKL/SKN

    return SKL, SKN, len(target) - SKL


def get_flow(path, src_ip, dst_ip):
    data = pd.read_csv(path)
    raw_data = data.values
    keywords = extract(i_url=path)[1:]

    print(path, "- len - ", len(raw_data))

    clean_data = []
    flows = []

    # 패킷의 전송 간격을 구함
    interval = []
    last = 0
    for line in raw_data:
        if line[2] == src_ip or line[2] == dst_ip:
            interval.append(line[1]-last)
            last = line[1]
            clean_data.append(line)

    # 패킷 평균 전송 간격
    avg = sum(interval) / len(interval)
    print(avg)

    diverge = []

    i = 0
    # 평균 전송간격보다 큰 경우 flow를 분리를 위한 인덱스 찾기
    for t in interval:
        if t >= avg:
            diverge.append(i)
        i = i + 1

    # 찾은 인덱스로 flow를 분리
    prefix = 0
    for idx in diverge:
        if idx >= len(clean_data):
            break
        flow = clean_data[prefix:idx]
        flows.append(flow)
        prefix = idx + 1

    print("## 총 ", len(flows), "개의 flow 발견")

    # 평균 인터벌  use
    avg_ivl = []
    # static keyword number average
    avg_skn = []
    # static keyword length average
    avg_skl = []
    # field size average
    avg_fsa = []
    # 평균 페이로드 크기 use
    avg_pay = []

    """
    # 평균 패킷 수
    avg_cnt = []
    # 프로토콜 넘뻐
    pcl = []
    """

    for flow in flows:
        last = 0
        ivl = 0
        payload = 0
        skl = 0
        skn = 0
        fsa = 0
        if len(flow) <= 0:
            continue
        for line in flow:
            tskl, tskn, tfsa = dynamic_cal(keywords, line[6])
            skl += tskl
            skn += tskn
            fsa += tfsa
            ivl = ivl + (line[1]-last)
            last = line[1]
            payload = payload + line[5]

        # 값 입력
        avg_ivl.append(ivl / len(flow))
        avg_pay.append(payload / len(flow))
        avg_skl.append(skl / len(flow))
        avg_skn.append(skn / len(flow))
        avg_fsa.append(fsa / len(flow))
        """
        avg_cnt.append(len(flow))
        if flow[0][4] == "UDP":
            pcl.append(17)
        elif flow[0][4] == "TCP":
            pcl.append(6)
        """
    print(sum(avg_pay)/len(avg_pay))

    return (np.array([avg_skn, avg_skl, avg_pay, avg_fsa, avg_ivl])).T   # [avg_ivl, avg_cnt, avg_pay, pcl]


def membership_function(data=pd.DataFrame([]), cols="", sigma=1.0):

    # find bound values from traffic flows in same traffic type data
    # find bound values from traffic flows in same traffic type data

    lbl = ""
    if cols == "CON":
        x_con = np.arange(0, 100.5, 0.5)

        conf_mal = fuzz.trapmf(x_con, [0, 20, 40, 60])
        conf_nor = fuzz.trapmf(x_con, [40, 60, 80, 100])
        """
        plt.plot(x_con, conf_mal, 'r', linewidth=1.5, label='mal')
        plt.plot(x_con, conf_nor, 'g', linewidth=1.5, label='nor')

        plt.show()
        """

        return [x_con, conf_mal, conf_nor]
    elif cols == "SKN":
        lbl = "Static Keyword Number"
    elif cols == "SKL":
        lbl = "Static Keyword Length"
    elif cols == "PSA":
        lbl = "Packet Size Average"
    elif cols == "FSA":
        lbl = "Field Size Average"
    else:
        lbl = "Packet Interval Average"

    print(data[cols].min(), "~", data[cols].max())

    chat_pia = data[cols]

    """
    # for quantile method
    # divide data
    chat_pia = chat_pia.sort_values()
    low_data = chat_pia[chat_pia <= np.quantile(chat_pia, 0.25)].to_numpy()  # 1-quartile
    match_data = chat_pia[chat_pia <= np.quantile(chat_pia, 0.75)][chat_pia >= np.quantile(chat_pia, 0.25)].to_numpy()
    high_data = chat_pia[chat_pia > np.quantile(chat_pia, 0.75)].to_numpy()  # 3-quartile
    """

    min = chat_pia.min()
    mean = chat_pia.mean()
    median = chat_pia.median()
    max = chat_pia.max()
    st = chat_pia.std()

    chat_pia = chat_pia.append(pd.Series([min - (st * sigma), max + (st * sigma)]))

    raw_data = (chat_pia.sort_values()).to_numpy()

    """
    fig, ((ax0, ax1, ax4), (ax3, ax2, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(24, 10))
    
    # Gaussian with 1-quantile average 3-quantile
    low = fuzz.gaussmf(low_data, low_data.mean(), low_data.std())
    match = fuzz.gaussmf(match_data, match_data.mean(), match_data.std())
    high = fuzz.gaussmf(high_data, high_data.mean(), high_data.std())

    ax0.plot(low_data, low, 'b', linewidth=1.5, label='Low')
    ax0.plot(match_data, match, 'g', linewidth=1.5, label='Match')
    ax0.plot(high_data, high, 'r', linewidth=1.5, label='High')
    ax0.set_title('quartile-based gaussian')
    ax0.set_xlabel(lbl)
    ax0.set_ylabel('Estimated Membership Degree')
    ax0.legend()
    """

    # Gaussian with min average max
    low_6 = fuzz.gaussmf(raw_data, min, st)
    match_6 = fuzz.gaussmf(raw_data, mean, st)
    high_6 = fuzz.gaussmf(raw_data, max, st)

    """
    ax3.plot(raw_data, low_6, 'b', linewidth=1.5, label='Low')
    ax3.plot(raw_data, match_6, 'g', linewidth=1.5, label='Match')
    ax3.plot(raw_data, high_6, 'r', linewidth=1.5, label='High')
    ax3.set_title('min/max-based gaussian')
    ax3.set_xlabel(lbl)
    ax3.set_ylabel('Estimated Membership Degree')
    ax3.legend()

    
    # Triangular with min, max and average
    low_2 = fuzz.trimf(raw_data, [min, min, mean])
    match_2 = fuzz.trimf(raw_data, [min, mean, max])  # average
    high_2 = fuzz.trimf(raw_data, [mean, max, max])

    ax1.plot(raw_data, low_2, 'b', linewidth=1.5, label='Low')
    ax1.plot(raw_data, match_2, 'g', linewidth=1.5, label='Match')
    ax1.plot(raw_data, high_2, 'r', linewidth=1.5, label='High')
    ax1.set_title('average triangular')
    ax1.set_xlabel(lbl)
    ax1.set_ylabel('Estimated Membership Degree')
    ax1.legend()

    # Triangular with min, max and median
    low_3 = fuzz.trimf(raw_data, [min, min, median])
    match_3 = fuzz.trimf(raw_data, [min, median, max])  # average
    high_3 = fuzz.trimf(raw_data, [median, max, max])

    ax2.plot(raw_data, low_3, 'b', linewidth=1.5, label='Low')
    ax2.plot(raw_data, match_3, 'g', linewidth=1.5, label='Match')
    ax2.plot(raw_data, high_3, 'r', linewidth=1.5, label='High')
    ax2.set_title('median triangular')
    ax2.set_xlabel(lbl)
    ax2.set_ylabel('Estimated Membership Degree')
    ax2.legend()

    # Trapezoidal with min, max and mean
    low_l = sorted([min * 0.8, min, min * 1.2, mean])
    match_l = sorted([min, mean * 0.8, mean * 1.2, max])
    high_l = sorted([mean, max * 0.8, max, max * 1.2])

    low_4 = fuzz.trapmf(raw_data, low_l)
    match_4 = fuzz.trapmf(raw_data, match_l)  # average
    high_4 = fuzz.trapmf(raw_data, high_l)

    ax4.plot(raw_data, low_4, 'b', linewidth=1.5, label='Low')
    ax4.plot(raw_data, match_4, 'g', linewidth=1.5, label='Match')
    ax4.plot(raw_data, high_4, 'r', linewidth=1.5, label='High')
    ax4.set_title('mean Trapezoidal')
    ax4.set_xlabel(lbl)
    ax4.set_ylabel('Estimated Membership Degree')
    ax4.legend()

    # Trapezoidal with min, max and median
    low_l = sorted([min * 0.8, min, min * 1.2, median])
    match_l = sorted([min, median * 0.8, median * 1.2, max])
    high_l = sorted([median, max * 0.8, max, max * 1.2])

    low_5 = fuzz.trapmf(raw_data, low_l)
    match_5 = fuzz.trapmf(raw_data, match_l)  # average
    high_5 = fuzz.trapmf(raw_data, high_l)

    ax5.plot(raw_data, low_5, 'b', linewidth=1.5, label='Low')
    ax5.plot(raw_data, match_5, 'g', linewidth=1.5, label='Match')
    ax5.plot(raw_data, high_5, 'r', linewidth=1.5, label='High')
    ax5.set_title('median Trapezoidal')
    ax5.set_xlabel(lbl)
    ax5.set_ylabel('Estimated Membership Degree')
    ax5.legend()

    plt.show()
    """

    return [low_6, match_6, high_6, raw_data]


def make_membership(data=pd.DataFrame([]), sigma=1.0):
    mf_skn = membership_function(data=data, cols="SKN", sigma=sigma)
    mf_skl = membership_function(data=data, cols="SKL", sigma=sigma)
    mf_psa = membership_function(data=data, cols="PSA", sigma=sigma)
    mf_fsa = membership_function(data=data, cols="FSA", sigma=sigma)
    mf_pia = membership_function(data=data, cols="PIA", sigma=sigma)

    return [mf_skn, mf_skl, mf_psa, mf_fsa, mf_pia]

def fuzzy_inference_engine(skn, skl, psa, fsa, pia, mf, con_mf):
    # 4.49, 3.64, 90, 85, 250
    x1, x2, x3, x4, x5 = (skn, skl, psa, fsa, pia)  # SKN, SKL, PSA, FSA, PIA input

    try:
        skn_lo = fuzz.interp_membership(mf[0][3], mf[0][0], x1)
        skn_mt = fuzz.interp_membership(mf[0][3], mf[0][1], x1)
        skn_hi = fuzz.interp_membership(mf[0][3], mf[0][2], x1)

        skl_lo = fuzz.interp_membership(mf[1][3], mf[1][0], x2)
        skl_mt = fuzz.interp_membership(mf[1][3], mf[1][1], x2)
        skl_hi = fuzz.interp_membership(mf[1][3], mf[1][2], x2)

        psa_lo = fuzz.interp_membership(mf[2][3], mf[2][0], x3)
        psa_mt = fuzz.interp_membership(mf[2][3], mf[2][1], x3)
        psa_hi = fuzz.interp_membership(mf[2][3], mf[2][2], x3)

        fsa_lo = fuzz.interp_membership(mf[3][3], mf[3][0], x4)
        fsa_mt = fuzz.interp_membership(mf[3][3], mf[3][1], x4)
        fsa_hi = fuzz.interp_membership(mf[3][3], mf[3][2], x4)

        pia_lo = fuzz.interp_membership(mf[4][3], mf[4][0], x5)
        pia_mt = fuzz.interp_membership(mf[4][3], mf[4][1], x5)
        pia_hi = fuzz.interp_membership(mf[4][3], mf[4][2], x5)

        hi = [skn_hi, skl_hi, psa_hi, fsa_hi, pia_hi]
        mt = [skn_mt, skl_mt, psa_mt, fsa_mt, pia_mt]
        lo = [skn_lo, skl_lo, psa_lo, fsa_lo, pia_lo]

        # if pia, skl, skn, psa, fsa is match then confidence is near 90
        rule1 = reduce(lambda x, y: x * y, mt)
        con_nor = np.fmin(np.multiply(rule1, 0.99), con_mf[2])

        # if pia, skl, skn, psa, fsa is high then confidence or pia, skl, skn, psa, fsa is low is near 10
        rule2 = np.fmax(reduce(lambda x, y: x * y, hi), reduce(lambda x, y: x * y, lo))
        con_mal = np.fmin(np.multiply(rule2, 0.01), con_mf[1])

        """
        # Visualize this
        fig, ax0 = plt.subplots(figsize=(8, 8))

        con0 = np.zeros_like(con_mf[0])
        ax0.fill_between(con_mf[0], con0, con_nor, facecolor='b', alpha=0.7)
        ax0.plot(con_mf[0], con_mf[1], 'b', linewidth=0.5, linestyle='--', )
        ax0.fill_between(con_mf[0], con0, con_mal, facecolor='g', alpha=0.7)
        ax0.plot(con_mf[0], con_mf[2], 'g', linewidth=0.5, linestyle='--')
        ax0.set_title('Output membership activity')
    
        # Turn off top/right axes
        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()

        plt.show()
     
        aggregate = np.fmax(con_mal, con_nor)
        print("CONFIDENCE(CENTROID)", fuzz.defuzz(con_mf[0], aggregate, 'centroid'))
        print("CONFIDENCE(MEAN)", fuzz.defuzz(con_mf[0], aggregate, 'mom'))
        print("CONFIDENCE(BISECTOR)", fuzz.defuzz(con_mf[0], aggregate, 'bisector'))
        print("CONFIDENCE(MIN)", fuzz.defuzz(con_mf[0], aggregate, 'som'))
        print("CONFIDENCE(MAX)", fuzz.defuzz(con_mf[0], aggregate, 'lom'))
        """

        # WEIGHTED AVERAGE DEFUZZIFICATION METHOD
        cf_1 = fuzz.defuzz(con_mf[0], con_mal, 'centroid')
        cf_2 = fuzz.defuzz(con_mf[0], con_nor, 'centroid')

        confidence = np.divide(max(con_mal) * cf_1 + max(con_nor) * cf_2, max(con_mal) + max(con_nor))
        #print("CONFIDENCE(WEIGHTED AVERAGE)", confidence)

        # conf_activation = fuzz.interp_membership(con_mf[0], np.fmax(cf_1, cf_2), confidence)

        """
        fig1, ax1 = plt.subplots(figsize=(8, 8))

        ax1.plot(con_mf[0], con_mf[1], 'b', linewidth=0.5, linestyle='--', )
        ax1.plot(con_mf[0], con_mf[2], 'g', linewidth=0.5, linestyle='--')
        ax1.fill_between(con_mf[0], con0, aggregate, facecolor='Orange', alpha=0.7)
        ax1.plot([confidence, confidence], [0, conf_activation], 'k', linewidth=1.5, alpha=0.9)
        ax1.set_title('Aggregated membership and result (line)')

        # Turn off top/right axes
        for ax in (ax1,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()
        """
    except:
        return 0

    return confidence

if __name__ == "__main__":
    
    # SYN Flood
    nor_data = pd.DataFrame(get_flow("./datasets/tcpSYNflood5m-1h_.csv", '172.27.224.250', '172.27.224.251'),
                            columns=["SKN", "SKL", "PSA", "FSA", "PIA"])

    ab_data = pd.DataFrame(get_flow("./datasets/tcpSYNflood5m-1h_ab.csv", '172.27.224.250', '172.27.224.251'),
                             columns=["SKN", "SKL", "PSA", "FSA", "PIA"])
    """
    # Ping Flood
    nor_data = pd.DataFrame(get_flow("./datasets/pingFlood5m-1h_.csv", '172.27.224.250', '172.27.224.251'),
                            columns=["SKN", "SKL", "PSA", "FSA", "PIA"])

    ab_data = pd.DataFrame(get_flow("./datasets/pingFlood5m-1h_ab.csv", '172.27.224.250', '172.27.224.251'),
                             columns=["SKN", "SKL", "PSA", "FSA", "PIA"])
    
    
    # Quary Flood
    nor_data = pd.DataFrame(get_flow("./datasets/querflooding5m-1h_.csv", '172.27.224.250', '172.27.224.251'),
                            columns=["SKN", "SKL", "PSA", "FSA", "PIA"])

    ab_data = pd.DataFrame(get_flow("./datasets/querflooding5m-1h_ab.csv", '172.27.224.250', '172.27.224.251'),
                             columns=["SKN", "SKL", "PSA", "FSA", "PIA"])   # SYN Flood
    """
    
    import time
    start = time.time() # learning start

    mf = make_membership(data=nor_data, sigma=1)
    con_mf = membership_function(data=nor_data, cols="CON")

    end = time.time()   # learning end
    print("Learning Timne", end - start)


    # Experiment
    print("\nChatting Flow ---------------------------------------------------------")
    nor_count = 0
    nor_conf = []
    nor_len = nor_data.values.__len__()
    for flow in nor_data.values:
        conf = fuzzy_inference_engine(flow[0], flow[1], flow[2], flow[3], flow[4], mf, con_mf)
        if conf > 0:
            nor_conf.append(conf)
            if conf > 65:
                nor_count += 1
    print("Declare - ", nor_count)
    print("Rate - ", nor_count/nor_len)
    print("Confidence - ", np.average(nor_conf))

    print("\nLow Flow ---------------------------------------------------------")
    ab_count = 0
    ab_conf = []
    ab_len = ab_data.values.__len__()
    for flow in ab_data.values:
        conf = fuzzy_inference_engine(flow[0], flow[1], flow[2], flow[3], flow[4], mf, con_mf)
        if conf > 0:
            ab_conf.append(conf)
            if conf > 65:
                ab_count += 1
    print("Declare - ", ab_count)
    print("Rate - ", ab_count / ab_len)
    print("Confidence - ", np.average(ab_conf))



    Precision = nor_count / (nor_count + ab_count)
    Recall = nor_count / nor_len
    Accuracy = (nor_count + (ab_len - ab_count)) / (nor_len + ab_len)
    F1_Score = 2 * ((Precision * Recall) / (Precision + Recall))
    print("PP - ", (nor_count / nor_len))
    print("FP - ", (ab_count / ab_len))
    print("Precision - ", Precision)
    print("Recall - ", Recall)
    print("Accuracy - ", Accuracy)
    print("F1_Score - ", F1_Score)

    f = open("./result/result.txt", "a+")
    f.write("PP - " + str((nor_count / nor_len)) + "\n")
    f.write("FP - " + str((ab_count / ab_len)) + "\n")
    f.write("Precision - " + str(Precision) + "\n")
    f.write("Recall - " + str(Recall) + "\n")
    f.write("Accuracy - " + str(Accuracy) + "\n")
    f.write("F1_Score - " + str(F1_Score) + "\n")

    f.close()

    """ threshold transition
    fp_rate = []
    threshold_list = np.linspace(50, 69.7, num=340)  # 60, 69.7, num=97

    for thr in threshold_list:
        chat_count = 0
        for cc in chat_conf:
            if cc > thr:
                chat_count += 1

        low_count = 0
        for lc in nor_conf:
            if lc > thr:
                low_count += 1

        high_count = 0
        for hc in ab_conf:
            if hc > thr:
                high_count += 1

        fp_rate.append((chat_count + high_count) / (chat_len + high_len))
        print("정탐률 - ", low_count / low_len)
        print("오탐률 - ", (chat_count + high_count) / (chat_len + high_len))
    """

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(np.arange(0, ab_conf.__len__(), step=1), ab_conf, 'b')
    ax1.set_title('Abnormal Traffic Confidence')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Estimated Confidence')

    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(np.arange(0, nor_conf.__len__(), step=1), nor_conf, 'g')
    ax2.set_title('Normal Traffic Confidence')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Estimated confidence')

    plt.show()
    """

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(threshold_list, fp_rate, 'c')
    ax4.set_title('FP-Rate Transition')
    ax4.set_xlabel('threshold')
    ax4.set_ylabel('False-Positive Rate')

    plt.show()
    """

    exit(0)
