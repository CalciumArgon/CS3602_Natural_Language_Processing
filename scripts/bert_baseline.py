# coding: utf-8

import sys, os, time, gc, json
from torch.optim import Adam
from tqdm import tqdm

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from torch.optim.lr_scheduler import MultiStepLR
from model.bert_baseline_model import BertSLU
from model.bert_fuse_model import FuseBertSLU

args = init_args(sys.argv[1:])

# time_stramp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
root_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_save_path = os.path.join(root_path, "checkpoints", args.name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path, exist_ok=True)

set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")
print("[Work on {}]".format(device))

start_time = time.time()
if not args.replace_place_name:
    train_path = os.path.join(args.dataroot, 'train.json')
else:
    train_path = os.path.join(args.dataroot, 'train_aug.json')

dev_path = os.path.join(args.dataroot, 'development.json')
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


if not args.fuse_chunk:
    model = BertSLU(args).to(device)
else:
    model = FuseBertSLU(args).to(device)
# ==========================================
model_name = f"{args.encoder_cell}-{args.decoder_cell}-lock{args.lock_bert_ratio}-replace{args.replace_place_name}.bin"
# ==========================================

if args.testing:
    # check_point = torch.load(open(args.ckpt, 'rb'), map_location=device)
    if args.ckpt is not None:
        check_point = torch.load(open(args.ckpt, 'rb'), map_location=device)
    else:
        check_point = torch.load(open(os.path.join(model_save_path, model_name), 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path")


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            # for j in range(len(current_batch)):
            #     if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
            #         print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example.load_dataset(test_path)
    predictions = {}
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=False)
            pred = model.decode(Example.label_vocab, current_batch)
            for pi, p in enumerate(pred):
                did = current_batch.did[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r', encoding='utf-8'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    scheduler = MultiStepLR(optimizer, milestones=args.decay_step, gamma=args.gamma)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, batch_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        trainbar = tqdm(range(0, nsamples, batch_size))
        for j, _ in enumerate(trainbar):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + batch_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            count += 1

            if j > 0 and j % 50 == 0:
                start_time = time.time()
                metrics, dev_loss = decode('dev')
                dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
                if args.ONE_EPOCH:
                    assert False, "One epoch done: {}".format((dev_loss,dev_acc,dev_fscore))
                # print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
                if dev_acc > best_result['dev_acc']:
                    best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
                    torch.save({
                        'epoch': i, 'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                    }, open(os.path.join(model_save_path, model_name), 'wb'))
                    print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))


                dev_info = {
                    "Best_Acc": best_result["dev_acc"],
                    "Dev_Acc": dev_acc,
                    "Dev_P": dev_fscore['precision'],
                    "Dev_R": dev_fscore['recall'],
                    "Dev_F": dev_fscore['fscore']
                }
                # for key, value in dev_info.items():
                #     writer.add_scalar(f"dev/{key}", value, j + i * 160)  # 160 = 「(nsamples / batch_size)

            trainbar.set_description(
                f"Epoch: {i} | L: {epoch_loss / count:.2f}| Best_Acc: {best_result['dev_acc']:.2f} | Acc: {dev_acc:.2f} | P: {dev_fscore['precision']:.2f} | R: {dev_fscore['recall']:.2f}| F: {dev_fscore['fscore']:.2f}"
            )
            # writer.add_scalar("train/epoch_loss", epoch_loss / count, j + i * 160)  # 160 = (nsamples / batch_size)
        
        torch.cuda.empty_cache()
        gc.collect()
    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))

else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
