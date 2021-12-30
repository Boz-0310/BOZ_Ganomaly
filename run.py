#%%
import os, warnings, argparse
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import source.datamanager as dman
import source.neuralnet as nn
import source.tf_process as tfp
import matplotlib.pyplot as plt

import source.custom_dataset

from sklearn.metrics import roc_curve, roc_auc_score, auc,f1_score
from sklearn.metrics import precision_score

tf.compat.v1.disable_v2_behavior()

AAA=source.custom_dataset.custom_data
global test_results

test_results=0

def main():
    global test_results
    dataset = dman.Dataset(AAA,normalize=FLAGS.datnorm)
    neuralnet = nn.SkipGANomaly(height=dataset.height, width=dataset.width, channel=dataset.channel,z_dim=FLAGS.z_dim, leaning_rate=FLAGS.lr)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=sess_config)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    con_list = tfp.training(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)
    test_results=tfp.test(sess=sess, neuralnet=neuralnet, saver=saver, dataset=dataset, batch_size=FLAGS.batch)
    cm,y_test,predictions,threshold,score_list,MSE_list = test_results

    plt.plot(con_list) # 畫con_loss

    np.savez('parameter.npz',height=dataset.height, width=dataset.width, channel=dataset.channel,z_dim=FLAGS.z_dim, leaning_rate=FLAGS.lr,threshold=threshold,max_val=dataset.max_val,min_val=dataset.min_val)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--z_dim', type=int, default=16, help='Dimension of latent vector')
    parser.add_argument('--lr', type=int, default=1e-5, help='Learning rate for training')#1e-5
    parser.add_argument('--epoch', type=int, default=3000, help='Training epoch')
    parser.add_argument('--batch', type=int, default=16, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()

# %%

cm,y_test,predictions,threshold,score_list,MSE_list = test_results
def draw_ROC_pic(fpr, tpr, auc1):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc1)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# print("confusion matrix:")
# print(cm)
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]

diagonal_sum = cm.trace()
sum_of_all_elements = cm.sum()
print("threshold: ", threshold)
print("Accuracy: ", diagonal_sum / sum_of_all_elements )
print("False Alarm Rate: ", FP/(FP+TP))
print("Leakage Rate: ", FN/(FN+TN))
print("precision_score: ",precision_score(y_test, predictions))
print("recall_score: ", TP/(TP+FN))
print("F1-Score: ", f1_score(y_test, predictions))

fpr, tpr, z = roc_curve(y_test, predictions)
auc1 = auc(fpr, tpr)
draw_ROC_pic(fpr, tpr, auc1)
print("Finish!")
#%%

# %%
plt.scatter(range(len(score_list)), score_list, c=['skyblue' if x == 1 else 'pink' for x in y_test])
plt.plot([0,len(score_list)],[threshold,threshold],c='red')

# %%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    fmt='d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plt.figure()
target_names='Mura','Normal'
plot_confusion_matrix(cm, classes=target_names,normalize=True,
                    title='confusion matrix')
plt.show()