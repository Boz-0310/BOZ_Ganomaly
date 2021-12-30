
from numpy.core.numeric import Inf, NaN
from sklearn.metrics import confusion_matrix
import os
import inspect
import time
import math
import shutil

from sklearn.metrics import roc_curve, roc_auc_score, auc


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
PACK_PATH = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))+"/.."


def make_dir(path):

    try:
        os.mkdir(path)
    except:
        pass


def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    if (gray.shape[2]>1):
        rgb[:, :, 0] = gray[:, :, 0]
        rgb[:, :, 1] = gray[:, :, 1]
        rgb[:, :, 2] = gray[:, :, 2]
    else:
        rgb[:, :, 0] = gray[:, :, 0]
        rgb[:, :, 1] = gray[:, :, 0]
        rgb[:, :, 2] = gray[:, :, 0]

    return rgb


def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try:
                tmp = data[x+(y*numd)]
            except:
                pass
            else:
                canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas


def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1, num_cont, i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" % (cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %
          (epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(
        PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="results")
    result_list = ["tr_resotring"]
    for result_name in result_list:
        make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(
        trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 20
    test_size = test_sq**2

    loss_con_list = [] #con loss 

    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(
            batch_size=test_size, fix=True)  # Initial batch
        x_restore = sess.run(neuralnet.x_hat,
                             feed_dict={neuralnet.x: x_tr, neuralnet.batch_size: x_tr.shape[0]})

        # save_img(contents=[x_tr, x_restore, (x_tr-x_restore)**2], \
        #     names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
        #     savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        while(True):
            # y_tr does not used in this prj.
            x_tr, y_tr, terminator = dataset.next_train(batch_size)

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries],
                                    feed_dict={
                                        neuralnet.x: x_tr, neuralnet.batch_size: x_tr.shape[0]},
                                    options=run_options, run_metadata=run_metadata)
            loss_enc, loss_con, loss_adv, loss_tot = sess.run([neuralnet.mean_loss_enc, neuralnet.mean_loss_con, neuralnet.mean_loss_adv, neuralnet.loss],
                                                              feed_dict={neuralnet.x: x_tr, neuralnet.batch_size: x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator):
                break

        print("Epoch [%d / %d] (%d iteration) Loss  Enc:%.3f, Con:%.3f, Adv:%.3f, Tot:%.3f"
              % (epoch, epochs, iteration, loss_enc, loss_con, loss_adv, loss_tot))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)
        loss_con_list.append(loss_con)
    return loss_con_list


def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        print("\nRestoring parameters")
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")
    try:
        shutil.rmtree("test")
    except:
        pass
    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list:
        make_dir(path=os.path.join("test", result_name))
    
    #為取得數量先抓一筆
    x_te = dataset.x_te[0:1]
    if(len(np.shape(x_te)) <= 3):
            x_te = np.expand_dims(x_te, axis=3)
    x_restore, score_anomaly = sess.run([neuralnet.x_hat, neuralnet.loss_enc],
                                            feed_dict={neuralnet.x: x_te, neuralnet.batch_size: x_te.shape[0]})
    [h, w, c] = x_restore[0].shape
    image_list=np.zeros([np.shape(dataset.y_te)[0],h,w,c])

    loss_list = []
    score_list=[]
    y_list = []
    testnum = 0
    while(True):
        # y_te does not used in this prj.
        x_te, y_te, terminator = dataset.next_test(1)

        x_restore, score_anomaly = sess.run([neuralnet.x_hat, neuralnet.loss_enc],
                                            feed_dict={neuralnet.x: x_te, neuralnet.batch_size: x_te.shape[0]})
        image_list[testnum,...]=x_restore      
        
        if math.isnan(score_anomaly):
            score_anomaly=np.array([9999.0],dtype='float32')
        score_list.append(score_anomaly)
        y_list.append(y_te.astype(int))
        if(y_te[0] == 1):
            loss_list.append(score_anomaly[0])
        testnum += 1
        if(terminator):
            break
    y_test = np.squeeze(np.array(y_list))
    auc_out, threshold = roc(y_test, score_list)

    loss_list = np.asarray(loss_list)
    loss_avg, loss_std = np.average(loss_list), np.std(loss_list)
    # outbound = loss_avg + (loss_std * 3)
    print("Loss  avg: %.5f, std: %.5f" % (loss_avg, loss_std))
    print("Outlier boundary: %.5f" % (threshold))
    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    # z_enc_tot, y_te_tot = None, None
    # loss4box = [[], [], [], [], [], [], [], [], [], []]
    temp_list = []

    # for i in range(np.shape(dataset.y_te)[0]):
    MSE_list = []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)
        outcheck = score_list[testnum] >= threshold
        fcsv.write("%d, %.5f, %r\n" % (y_test[testnum], score_list[testnum], outcheck))


        [h, w, c] = image_list[testnum].shape
        #原圖、生成、相減
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = image_list[testnum]
        canvas[:, w*2:, :] = (x_te[0]-image_list[testnum])**2
        
        MSE_list.append(np.sum(canvas[:, w*2:, :])/(h*w*c))
        
        #原圖、生成
        # canvas = np.ones((h, w*2, c), np.float32)
        # canvas[:, :w, :] = x_te[0]

        # canvas[:, w:w*2, :] = image_list[testnum]

        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%d-%05d.png" %
                                    (y_test[testnum],testnum)), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%d-%05d.png" %
                                    (y_test[testnum],testnum)), gray2rgb(gray=canvas))
        temp_list.append(outcheck.astype(int)*-1+1)
        testnum += 1
        if(terminator):
            break
    predictions = np.squeeze(np.array(temp_list))


    # while(True):
    #     # y_te does not used in this prj.
    #     x_te, y_te, terminator = dataset.next_test(1)

    #     x_restore, score_anomaly = sess.run([neuralnet.x_hat, neuralnet.loss_enc],
    #                                         feed_dict={neuralnet.x: x_te, neuralnet.batch_size: x_te.shape[0]})

    #     outcheck = score_anomaly >= outbound
    #     fcsv.write("%d, %.5f, %r\n" % (y_te, score_anomaly, outcheck))

    #     [h, w, c] = x_restore[0].shape
    #     canvas = np.ones((h, w*3, c), np.float32)
    #     canvas[:, :w, :] = x_te[0]
    #     canvas[:, w:w*2, :] = x_restore[0]
    #     canvas[:, w*2:, :] = (x_te[0]-x_restore[0])**2
    #     if(outcheck):
    #         plt.imsave(os.path.join("test", "outbound", "%08d.png" %
    #                                 (testnum)), gray2rgb(gray=canvas))
    #     else:
    #         plt.imsave(os.path.join("test", "inbound", "%08d.png" %
    #                                 (testnum)), gray2rgb(gray=canvas))

    #     testnum += 1
    #     temp_list.append(outcheck.astype(int)*-1+1)
    #     if(terminator):
    #         break
    # predictions = np.squeeze(np.array(temp_list))

    cm = confusion_matrix(y_true=y_test, y_pred=predictions)
    # cm=0
    test_results=(cm,y_test,predictions,threshold,score_list,MSE_list)
    return test_results 

    # boxplot(contents=loss4box, savename="test-box.png")

def roc(labels, scores):
    roc_auc = dict()
    fpr, tpr, threshold = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(abs(tpr - fpr))
    optimal_threshold = threshold[optimal_idx]
    return roc_auc, optimal_threshold

