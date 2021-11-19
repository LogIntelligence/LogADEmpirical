import sys

sys.path.extend(["../../", "../", "./"])
from hdbscan import HDBSCAN as dbscan
from logadempirical.PLELog.data.Embedding import *
from logadempirical.PLELog.approaches.hdbscan_options import *
from logadempirical.PLELog.approaches.RNN_pipeline import *
from logadempirical.PLELog.approaches.RNN_pipeline import train_model
import pickle, argparse, torch.optim.lr_scheduler
from logadempirical.PLELog.utils.ADHelper import *
from logadempirical.PLELog.utils.Config import *
from logadempirical.PLELog.model.GRU import *
from logadempirical.PLELog.data.DataLoader import *
from sklearn.decomposition import FastICA
import logging
import matplotlib as mpl

fixLength = 120
start = time.time()

vocab = None
refresh = False
random_state = 6


# random.seed(random_state)
# np.random.seed(random_state)


def record_data(dir, train, dev, test):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(os.path.join(dir, 'train'), 'w', encoding='utf-8') as writer:
        for instance in train:
            writer.write(str(instance) + '\n')
    if dev and len(dev) > 0:
        with open(os.path.join(dir, 'dev'), 'w', encoding='utf-8') as writer:
            for instance in dev:
                writer.write(str(instance) + '\n')
    with open(os.path.join(dir, 'test-hdfs'), 'w', encoding='utf-8') as writer:
        for instance in test:
            writer.write(str(instance) + '\n')


def prepare_data(logID2Temp, templateVocab, dataset, fixLength, logger, ratio=[6, 1, 3], shuffle=True, output_dir=""):
    start = time.time()
    if dataset == 'bgl' or dataset == "spirit" or dataset == "tdb" or dataset == "hdfs":
        train, val, test = load_from_structured(logID2Temp, fixLength, ratio, dataset, output_dir=output_dir)
    else:
        logger.error("Unknown dataset")
        exit(-2)

    if not isinstance(ratio, list):
        logger.error('Ratio parameter is wrong, list required.')
        exit(-1)
    if not len(ratio) == 3:
        logger.error(
            'Please make sure ratio list contains three parts, which represent the split ratio of train, dev and test-hdfs.')
        exit(-2)
    global labelledNum
    normalCount = 0
    abnormalCount = 0
    for instance in train:
        calRepr4Instance_nlp(instance, templateVocab)
        if instance.type == 'Normal':
            normalCount += 1
        else:
            abnormalCount += 1
    for instance in val:
        calRepr4Instance_nlp(instance, templateVocab)
        if instance.type == 'Normal':
            normalCount += 1
        else:
            abnormalCount += 1
    for instance in test:
        calRepr4Instance_nlp(instance, templateVocab)
        if instance.type == 'Normal':
            normalCount += 1
        else:
            abnormalCount += 1
    # print(test[0].repr)
    logger.info('normalCount ' + str(normalCount))
    logger.info('abnormalCount ' + str(abnormalCount))

    prepretrain = deepCopy(train)
    pre_dev = deepCopy(val)
    pre_test = deepCopy(test)

    numlabel = int(len(train) * 1)
    labelledInstances = []
    unlabelledInstances = []

    if shuffle:
        random.shuffle(prepretrain)

    for index in range(0, numlabel):
        if prepretrain[index].type == 'Normal':
            labelledInstances.append(prepretrain[index])
        else:
            unlabelledInstances.append(prepretrain[index])

    for index in range(numlabel, len(train)):
        unlabelledInstances.append(prepretrain[index])

    labelledNum = len(labelledInstances)
    logger.info('Number of labelled normal instances is : %d' % labelledNum)
    unlabelledNum = len(unlabelledInstances)
    logger.info('Number of unlabelled instances is : %d' % unlabelledNum)

    pre_train = deepCopy(labelledInstances)
    pre_train.extend(unlabelledInstances)

    logger.info('Train: %d, Dev: %d, Test: %d' % (len(pre_train), len(pre_dev), len(pre_test)))

    end = time.time()
    logger.info('Finish prepare dataset. time = %.2f' % (end - start))
    return pre_train, pre_dev, pre_test, labelledNum


def PULearn1(pre_train, pre_dev, pre_test, save_path, min_cluster_size=None, min_samples=None,
             option=None, rd=False, logger=None):
    global labelledNum, refresh
    start_time = time.time()
    if pre_train is not None and (not os.path.exists(os.path.join(save_path, 'HDBscan_result.txt')) or refresh):
        trainReprs = []
        for instance in pre_train:
            trainReprs.append(instance.repr)
        if rd != -1:
            logger.info('FastICA:')
            transformer = FastICA(n_components=rd)
            trainReprs = transformer.fit_transform(trainReprs)
            logger.info('Finished at %.2f' % (time.time() - start_time))
        estimator = dbscan(algorithm='best',
                           min_cluster_size=min_cluster_size if min_cluster_size else 5,
                           min_samples=min_samples if min_samples and min_samples != -1 else None,
                           core_dist_n_jobs=10,
                           metric='euclidean')
        predicts = estimator.fit_predict(np.asarray(trainReprs, dtype=np.float)).tolist()
        outliers = estimator.outlier_scores_.tolist()
        hdbscanInstances = []
        for i, inst in enumerate(pre_train):
            type = 'unlabelled'
            if i < labelledNum:
                type = 'labelled'
            hdbscanInstances.append(
                HDbscan_Instance(inst.src_events, inst.type, inst.repr, inst.src_blk, predicts[i],
                                 outliers[i], type))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'HDBscan_result.txt'), 'w', encoding='utf-8') as writer:
            for instance in hdbscanInstances:
                writer.write(str(instance) + '\n')
        logger.info('HDBscan training finish, time = %.3f' % (time.time() - start_time))
    else:
        instances, predicts, outliers = loadHDBscanResult(os.path.join(save_path, 'HDBscan_result.txt'))
        logger.info("load: " + str(os.path.join(save_path, 'HDBscan_result.txt')))

    if not option or option == 0:
        logger.info('Option 0')
        new_train, precision, recall, f, num_of_neg1 = option_0(predicts, pre_train, labelledNum, logger)
    else:
        logger.info('Option 1: confidence')
        new_train, precision, recall, f, num_of_neg1 = option_1(predicts, outliers, pre_train, labelledNum, logger)
    logger.info('HDBscan result clustering finish, time = %.3f' % (time.time() - start_time))
    end = time.time()
    logger.info('New dataset finishes. time = %.2f' % (end - start_time))
    record_data(save_path, new_train, pre_dev, pre_test)
    logger.info('PULearn finished. time = %.3f' % (time.time() - start_time))
    logger.info('Train : %d, Dev : %d, Test : %d' % (len(new_train), len(pre_dev), len(pre_test)))
    return new_train, pre_dev, pre_test, precision, recall, f, num_of_neg1, len(
        list(filter(lambda x: x == 0, outliers)))


def PULearn(pre_train, pre_dev, pre_test, save_path, min_cluster_size=None, min_samples=None,
            option=None, rd=False, logger=None):
    global labelledNum, refresh
    start_time = time.time()
    num_of_neg1 = 0
    if option == 6:
        refresh = True
    if not os.path.exists(os.path.join(save_path, 'HDBscan_result.txt')) or refresh:
        trainReprs = []
        for instance in pre_train:
            trainReprs.append(instance.repr)
        if rd != -1:
            logger.info('FastICA:')
            transformer = FastICA(n_components=rd)
            trainReprs = transformer.fit_transform(trainReprs)
            logger.info('Finished at %.2f' % (time.time() - start_time))

        estimator = dbscan(algorithm='best',
                           min_cluster_size=min_cluster_size if min_cluster_size else 5,
                           min_samples=min_samples if min_samples and min_samples != -1 else None,
                           core_dist_n_jobs=10)
        predicts = estimator.fit_predict(np.asarray(trainReprs, dtype=np.float)).tolist()
        outliers = estimator.outlier_scores_.tolist()
        hdbscanInstances = []
        for i, inst in enumerate(pre_train):
            type = 'unlabelled'
            if i < labelledNum:
                type = 'labelled'
            hdbscanInstances.append(
                HDbscan_Instance(inst.src_events, inst.type, inst.repr, inst.src_blk, predicts[i],
                                 outliers[i], type))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, 'HDBscan_result.txt'), 'w', encoding='utf-8') as writer:
            for instance in hdbscanInstances:
                writer.write(str(instance) + '\n')
        logger.info('HDBscan training finish, time = %.3f' % (time.time() - start_time))
    else:
        logger.info("load: " + str(os.path.join(save_path, 'HDBscan_result.txt')))
        instances, predicts, outliers = loadHDBscanResult(os.path.join(save_path, 'HDBscan_result.txt'))
    if not option or option == 0:
        logger.info('Option 0')
        new_train, precision, recall, f, num_of_neg1 = option_0(predicts, pre_train, labelledNum, logger)
    else:
        if option == 1:
            logger.info('Option 1: confidence')
            new_train, precision, recall, f, num_of_neg1 = option_1(predicts, outliers, pre_train, labelledNum, logger)
        else:
            new_train, precision, recall, f, num_of_neg1 = option_0(predicts, pre_train, labelledNum, logger)
    logger.info('HDBscan result clustering finish, time = %.3f' % (time.time() - start_time))
    end = time.time()
    logger.info('New dataset finishes. time = %.2f' % (end - start_time))
    record_data(save_path, new_train, pre_dev, pre_test)
    logger.info('PULearn finished. time = %.3f' % (time.time() - start_time))
    logger.info('Train : %d, Dev : %d, Test : %d' % (len(new_train), len(pre_dev), len(pre_test)))
    return new_train, pre_dev, pre_test, precision, recall, f, num_of_neg1, len(
        list(filter(lambda x: x == 0, outliers)))


def main_process(save_path, pre_train, pre_dev, pre_test, ratios, hdbscan_option,
                 min_samples, min_cluster_size, reduce_dim, config, extra_args, thread_num, target_gpu, logger,
                 threshold=0.5, dataset="BGL", logID2Temp={}):
    mpl.use('Agg')
    gpu = torch.cuda.is_available()
    mid_dir = 'train-' + str(ratios[0])
    save_path = os.path.join(save_path, mid_dir)
    min_samples = min_samples if min_samples != -1 else min_cluster_size
    postfix = 'rd-' + str(reduce_dim) + '_mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples) + '_random-' + str(
        random_state)
    save_path = os.path.join(save_path, postfix)
    logger.info('Save Hdbscan results in %s' % save_path)
    if hdbscan_option != -1:
        logger.info('Start HDBSCAN Training.')
        logger.info('Min_cluster_size=%d, min_samples = %d' % (min_cluster_size, min_samples))
        train, dev, test, precision, recall, f, num_of_neg1, num_outlier0 = PULearn(pre_train, pre_dev, pre_test,
                                                                                    save_path, min_cluster_size,
                                                                                    min_samples,
                                                                                    hdbscan_option,
                                                                                    rd=reduce_dim,
                                                                                    logger=logger)


    else:
        postfix = 'upperbound'
        save_path = os.path.join(save_path, postfix)
        logger.info('Upperbound.')
        record_data(save_path, pre_train, pre_dev, pre_test)
        train, dev, test = pre_train, pre_dev, pre_test
        precision, recall, f = 1, 1, 1
        num_of_neg1, num_outlier0 = 0, 0

    vocab = creatVocab(train)
    # logger.info('Load configfile %s' % config_file)
    # config = Configurable(config_file, extra_args)
    torch.set_num_threads(thread_num)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))
    config.use_cuda = False
    if gpu and target_gpu != -1:
        config.use_cuda = True
        torch.cuda.set_device(target_gpu)
        logger.info('GPU ID:' + str(target_gpu))
    logger.info("\nGPU using status: " + str(config.use_cuda))
    model = AttGRUModel(vocab, config, vec)
    if config.use_cuda:
        model = model.cuda(target_gpu)
    classifier = AnomalyDetectionBCELoss(model, vocab)
    outputFile = dataset + '_' + str(hdbscan_option) + '_mcs-' + str(min_cluster_size) + '_ms' + str(
        min_samples) + '_rd-' + str(reduce_dim) \
                 + '_hidden-' + str(config.lstm_hiddens) + '_layers-' + str(config.lstm_layers) + '.out'
    outputFile = os.path.join('output_res', outputFile)
    dev_p, dev_r, dev_f, final_p, final_r, final_f = train_model(train, dev, test, classifier, vocab, config,
                                                                 vec=vec, logger=logger, outputFile=outputFile,
                                                                 threshold=threshold)
    res = ','.join([str(hdbscan_option), dataset, str(len(train)), str(len(dev)), str(len(test)),
                    str(reduce_dim), str(min_cluster_size), str(min_samples), str(config.lstm_hiddens),
                    str(config.lstm_layers),
                    str(num_of_neg1), str(num_outlier0),
                    str(precision), str(recall), str(f),
                    str(final_p), str(final_r), str(final_f),
                    str(dev_p), str(dev_r), str(dev_f)
                    ])
    print(res)
    vocab = pickle.load(open(config.save_vocab_path, 'rb'))
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    res = evaluate_online(pre_test, config, vocab, logger, vec, outputFile=None, threshold=threshold,
                          id2tem=logID2Temp, dat=dataset)

    return res


def run_PLELog(options):
    upper = 1000
    step = 50

    ### gpu
    gpu = torch.cuda.is_available()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='./logadempirical/PLELog/config/{}.cfg',
                           help='Configuration file for Attention-Based GRU Network.')
    argparser.add_argument('--gpu', default=0, type=int, help='GPU ID if using cuda, -1 if cpu.')
    argparser.add_argument('--thread', default=1, type=int, help='Number of thread to use. Default value is 1')
    argparser.add_argument('--hdbscan_option', default=0, type=int,
                           help='Different strategies of HDBSCAN clustering. 0 for PLELog_noP, 1 for PLELog, -1 for upperbound.')
    # argparser.add_argument('--dataset', default='bgl', type=str, help='Choose dataset, bgl, hdfs, spirit, '
    #                                                                   'or tdb (thunderbird)')
    argparser.add_argument('--train_ratio', default=7, type=int, help='Ratio of train data.')
    argparser.add_argument('--dev_ratio', default=1, type=int, help='Ratio of dev data.')
    argparser.add_argument('--test_ratio', default=2, type=int, help='Ratio of test data.')
    argparser.add_argument('--min_cluster_size', default=100, type=int,
                           help='Minimum cluster size, a parameter of HDBSCAN')
    argparser.add_argument('--min_samples', default=-1, type=int, help='Minimum samples, a parameter of HDBSCAN')
    argparser.add_argument('--reduce_dim', default=50, type=int, help='Target dimension of FastICA')
    argparser.add_argument('--threshold', default=0.5, type=float,
                           help='Threshold for final classification, any instance with "anomalous score" higher than this threshold will be regarded as anomaly.')

    args, extra_args = argparser.parse_known_args()
    config_file = args.config_file.format(options['dataset_name'].upper())
    dataset = options['dataset_name']
    target_gpu = args.gpu
    ratios = [args.train_ratio, args.dev_ratio, args.test_ratio]

    # dataset = args.dataset
    hdbscan_option = args.hdbscan_option
    thread_num = args.thread
    min_samples = args.min_samples
    min_cluster_size = args.min_cluster_size
    reduce_dim = args.reduce_dim
    threshold = args.threshold
    config = Configurable(config_file, extra_args, options)
    # Specify logger
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger_name = '_'.join(
        [dataset, str(hdbscan_option), 'rd-' + str(reduce_dim),
         'mcs-' + str(min_cluster_size),
         'ms-' + str(min_samples),
         'hidden_' + str(config.lstm_hiddens),
         'layer_' + str(config.lstm_layers)])

    hdlr = logging.FileHandler(os.path.join('logs', logger_name))
    logger = logging.getLogger('main')
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    if dataset == "hdfs" or dataset == "bgl" or dataset == "spirit" or dataset == "tdb":
        templatesDir = options['data_dir']# + dataset
        save_path = options['data_dir']# + dataset
        print(templatesDir, logger)
        logID2Temp, templates = load_templates_from_structured(templatesDir, logger, dataset,
                                                               log_file=options['log_file'])
        templateVocab = nlp_emb_mergeTemplateEmbeddings_BGL(save_path, templates, dataset, logger)
        pre_train, pre_dev, pre_test, _ = prepare_data(logID2Temp, templateVocab, dataset, fixLength, logger, ratios,
                                                       output_dir=options['output_dir'])
    else:
        logger.info("Unknown dataset")
        exit(-2)

    print("Start training...")
    train_save_path = options['output_dir'] + "plelog/"
    result = main_process(save_path=train_save_path, pre_train=pre_train, pre_dev=pre_dev, pre_test=pre_test,
                          ratios=ratios,
                          hdbscan_option=hdbscan_option,
                          min_samples=min_samples,
                          min_cluster_size=min_cluster_size, reduce_dim=reduce_dim,
                          config=config, extra_args=extra_args, thread_num=thread_num, target_gpu=target_gpu,
                          logger=logger,
                          threshold=threshold, dataset=dataset, logID2Temp=logID2Temp)
    print("Precision: {}, Recall: {}, Specificity: {}, F1: {}".format(result[0], result[1], result[2], result[3]))
