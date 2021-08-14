import matplotlib.pyplot as plt

with open("model/absa_bertcrf/history.txt", "r", encoding="utf-8") as fr:
    history_bertcrf = fr.read()
    history_bertcrf = eval(history_bertcrf)

with open("model/absa_bertlinear/history.txt", "r", encoding="utf-8") as fr:
    history_bertlinear = fr.read()
    history_bertlinear = eval(history_bertlinear)

with open("model/absa_bertgru/history.txt", "r", encoding="utf-8") as fr:
    history_bertgru = fr.read()
    history_bertgru = eval(history_bertgru)

plt.subplot(221)
plt.plot(history_bertcrf["loss"])
plt.plot(history_bertlinear["loss"])
plt.plot(history_bertgru["loss"])
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['CRF', 'Linear','GRU'])

plt.subplot(222)
plt.plot(history_bertcrf["crf_acc"])
plt.plot(history_bertlinear["acc"])
plt.plot(history_bertgru["acc"])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['CRF', 'Linear','GRU'])

plt.subplot(223)
plt.plot(history_bertcrf["val_loss"])
plt.plot(history_bertlinear["val_loss"])
plt.plot(history_bertgru["val_loss"])
plt.title('val_loss')
plt.ylabel('val_loss')
plt.xlabel('Epoch')
plt.legend(['CRF', 'Linear','GRU'])

plt.subplot(224)
plt.plot(history_bertcrf["val_crf_acc"])
plt.plot(history_bertlinear["val_acc"])
plt.plot(history_bertgru["val_acc"])
plt.title('val_acc')
plt.ylabel('val_acc')
plt.xlabel('Epoch')
plt.legend(['CRF', 'Linear','GRU'])

plt.suptitle("Model Metrics")

plt.tight_layout()
plt.savefig("compare.jpg", dpi=500, bbox_inches="tight")

# plt.show()
