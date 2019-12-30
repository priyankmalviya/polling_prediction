import densenet as dnet
import inceptionnet as inet
import mobilenet as mnet
import nasnetlarge as nnet_L
import nasnetmobile as nnet_M
import vgg16 as vgg
import xceptionnet as xception
import helpers

import pandas as pd
pd.set_option('display.max_columns', 50)
BATCH_SIZE = 64
PATH = '/content/test_images'

def merge_dataframe(df1,df2,col):
    df = pd.merge(df1,df2, on = col)
    return df


def get_dnet_result():
    df_dnet,acc_dnet = dnet.test(BATCH_SIZE,224,PATH,'/content/models/densenet.h5')
    return df_dnet,acc_dnet

def get_inet_result():
    df_inet,acc_inet = inet.test(BATCH_SIZE,299,PATH,'/content/models/inet.h5')
    return df_inet,acc_inet

def get_mnet_result():
    df_mnet,acc_mnet = mnet.test(BATCH_SIZE,224,PATH,'/content/models/mnet.h5')
    return df_mnet,acc_mnet

def get_nnetL_result():
    df_nnet_L,acc_nnet_L = nnet_L.test(BATCH_SIZE,331,PATH,'/content/models/nnet_L.h5')
    return df_nnet_L, acc_nnet_L

def get_nnetM_result():
    df_nnet_M,acc_nnet_M = nnet_M.test(BATCH_SIZE,224,PATH,'/content/models/nnet_M.h5')
    return df_nnet_M, acc_nnet_M

def get_vgg_result():
    df_vgg,acc_vgg = vgg.test(BATCH_SIZE,224,PATH,'/content/models/vgg.h5')
    return df_vgg,acc_vgg

def get_xception_result():
    df_xception,acc_xception = xception.test(BATCH_SIZE,299,PATH,'/content/models/xception.h5')
    return df_xception,acc_xception

def get_poll_prediction(df):
    for i ,row in df.iterrows():
        agg_dic = {}
        if row['dnet'] not in agg_dic.keys():
            agg_dic[row['dnet']] = 1
        else:
            agg_dic[row['dnet']] += 1

        if row['inet'] not in agg_dic.keys():
            agg_dic[row['inet']] = 1
        else:
            agg_dic[row['inet']] += 1

        if row['vgg'] not in agg_dic.keys():
            agg_dic[row['vgg']] = 1
        else:
            agg_dic[row['vgg']] += 1

        if row['xception'] not in agg_dic.keys():
            agg_dic[row['xception']] = 1
        else:
            agg_dic[row['xception']] += 1

        if row['mnet'] not in agg_dic.keys():
            agg_dic[row['mnet']] = 1
        else:
            agg_dic[row['mnet']] += 1

        if row['nnet_L'] not in agg_dic.keys():
            agg_dic[row['nnet_L']] = 1
        else:
            agg_dic[row['nnet_L']] += 1

        if row['nnet_M'] not in agg_dic.keys():
            agg_dic[row['nnet_M']] = 1
        else:
            agg_dic[row['nnet_M']] += 1

        #sorting the dictionary and getting the most voted prediction
        final = list({k: v for k, v in sorted(agg_dic.items(), reverse = True, key=lambda item: item[1])}.keys())[0]
        df.loc[df['image'] == row['image'],'Final'] = final
    return df

def to_list(df,col):
    return list(df[col])


def get_poll_accuracy(df):
    clss_lst = to_list(df,'class')
    final_lst = to_list(df,'Final')
    if len(clss_lst) == len(final_lst):
        count = 0
        for i in range(0,len(clss_lst)):
            orig_clss = clss_lst[i]
            final_clss = final_lst[i]
            if orig_clss.lower() == final_clss.lower():
                count += 1
    else:
        print('Mismatch in total number of records')
    return ((count/len(clss_lst))*100)



if __name__ == '__main__':
    df1,acc_dnet = get_dnet_result()

    df2,acc_inet = get_inet_result()
    df1 = merge_dataframe(df1,df2,'image')

    df2 , acc_mnet = get_mnet_result()
    df1 = pd.merge(df1,df2, on = 'image')

    df2, acc_nnet_L = get_nnetL_result()
    df1 = pd.merge(df1,df2, on = 'image')

    df2, acc_nnet_M = get_nnetM_result()
    df1 = pd.merge(df1,df2, on = 'image')

    df2,acc_vgg = get_vgg_result()
    df1 = pd.merge(df1,df2, on = 'image')

    df2, acc_xception = get_xception_result()
    df1 = pd.merge(df1,df2, on = 'image')

    df1 = get_poll_prediction(df1)


    print('******** FINAL DF **********')
    print(df1.tail(10))

    print('****** DNET ***********')
    print(acc_dnet)

    print('****** INET ***********')
    print(acc_inet)

    print('****** MNET ***********')
    print(acc_mnet)

    print('****** NNET_L ***********')
    print(acc_nnet_L)

    print('****** NNET_M ***********')
    print(acc_nnet_M)

    print('****** VGG ***********')
    print(acc_vgg)

    print('****** XCEPTION ***********')
    print(acc_xception)


    acc_poll = get_poll_accuracy(df1)
    print('****** FINAL ***********')
    print(acc_poll)
