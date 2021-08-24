#import asr_service_v2 as asr_service #load model
#import recorder
import pandas as pd
import numpy as np 
import asr_service_v2 as asr_service #load model
import re


"""---------------參數配置------------------"""
#注意：有任何檔案地址的部分都要記得更改成自己電腦檔案的地址
corpus_dir='C:/Users/Gene/Desktop/asr_wav2vec2-master-c3f9f0c267a23664f51ed5277d0037b5c363cefd/scripts/zh-TW_cer'
wav_dir='C:/Users/Gene/Desktop/asr_wav2vec2-master-c3f9f0c267a23664f51ed5277d0037b5c363cefd/scripts/zh-TW_cer/clips'
"""---------------------------------------"""

sum_Levenshtein_distance = 0
sum_total_words = 0
chars_to_ignore_regex = r"[¥•＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·'℃°•·．﹑︰〈〉─《﹖﹣﹂﹁﹔！？｡。＂＃＄％＆＇（）＊＋，﹐－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.．!\"#$%&()*+,\-.\:;<=>?@\[\]\\\/^_`{|}~]"

def CER(hypothesis,reference):
        #標準化
        Levenshtein_distance = 0
        total_words = 0
        d = np.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=np.uint8)#生成空矩陣d 放置解答與預測的資料
        d = d.reshape((len(reference) + 1, len(hypothesis) + 1))
        for i in range(len(reference) + 1):
            for j in range(len(hypothesis) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i
     
        # 比較文字
        for i in range(1, len(reference) + 1):
            for j in range(1, len(hypothesis) + 1):
                if reference[i - 1] == hypothesis[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1 #替換 S
                    insertion = d[i][j - 1] + 1 #插入 I
                    deletion = d[i - 1][j] + 1 #刪除 D
                    d[i][j] = min(substitution, insertion, deletion)
                    
        Levenshtein_distance = d[len(reference)][len(hypothesis)] #萊文斯坦距離，指兩個字串之間，由一個轉成另一個所需的最少編輯操作次數。
        total_words = int(len(reference))

        return Levenshtein_distance,total_words




if __name__ =="__main__":
    print("模型載入完畢")
    raw_data = pd.read_csv(corpus_dir + '/test.tsv',sep='\t')
    raw_data = raw_data.values
    data_path = raw_data[:,1]    #[第幾筆資料,資料的參數ex.路徑、測試者年齡與性別][資料中的第幾個字]
    data_label = raw_data[:,2]
    print(("總共有"+  str(len(data_path)) +"筆資料，請問要跑幾到幾筆的數據: "))
    start = input("起始:")
    end = input("結束:")
    try:    #預防檔案寫入發生錯誤導致要重跑數據
        with open('測試_test_cer_辨識結果_1-1240筆資料.txt', 'w+', encoding='UTF-8') as file:
            for n in range(int(start)-1,int(end)):    #自訂義執行語料文檔中的哪幾筆資料
                print("執行第"+str(n+1)+"筆資料")
                # print(str(n+1))
                data=asr_service.load_file_to_data(wav_dir+'/'+data_path[n] +'.mp3.wav')  #將音檔丟入模型辨識
                reference = data_label[n]    # reference 表示解答數據
                reference = re.sub(chars_to_ignore_regex, "", reference)
                hypothesis = asr_service.predict(data)    # hypothesis 表示預測數據          
                hypothesis = list(hypothesis[0])    #將預測數據存入list以便於計算CER
                print(reference)    #顯示label
                print(hypothesis)    #顯示預測數據
                Levenshtein_distance,total_words = CER(hypothesis,reference)    #比較字串獲得CER
                sum_Levenshtein_distance += Levenshtein_distance    #加總萊文斯坦距離
                sum_total_words += total_words    #加總解答的字數
                hypothesis = "".join(hypothesis)    #將預測數據由list轉成字串
                file.write(str(data_path[n])+"\t"+ str(reference) +"\t["+ str(hypothesis) +"]\n")    #將結果寫入檔案
                print("")
            file.write(str("Levenshtein_distance : "+ str(sum_Levenshtein_distance) + "\t total_words : \t"+ str(sum_total_words) + "\t\n")) #將結果寫入檔案
            file.write('total_cer : {:.10%}'.format(sum_Levenshtein_distance/sum_total_words)) #將結果寫入檔案
            print('total_cer : {:.10%}'.format(sum_Levenshtein_distance/sum_total_words))
            print(sum_Levenshtein_distance)
            print(sum_total_words) 
    except:
        print("寫入文檔發生問題!!!")
        print('total_cer : {:.10%}'.format(sum_Levenshtein_distance/sum_total_words))
        print("目前文檔的總Levenshtein_distance : "+ str((sum_Levenshtein_distance)))
        print("目前文檔的總字數 : "+str((sum_total_words)))
        print("請自行將結果寫入文檔中")