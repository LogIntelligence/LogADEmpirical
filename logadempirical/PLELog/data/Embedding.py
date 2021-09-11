from logadempirical.PLELog.data.DataLoader import *




def deepCopy(source):
    res = []
    for item in source:
        res.append(item)
    return res


def calRepr4Instance_nlp_BGL(instance, templateVocab):
    for k, v in templateVocab.items():
        embedSize = len(v)
        break
    assert embedSize == 300
    placeHolder = np.zeros(embedSize)
    if len(instance.src_events) > 0:
        for event in instance.src_events:
            placeHolder += templateVocab[event]
    instance.setSimpleRepr(placeHolder)


def calRepr4Instance_nlp(instance, templateVocab):
    for k, v in templateVocab.items():
        embedSize = len(v)
        break
    assert embedSize == 300
    placeHolder = np.zeros(embedSize)
    for event in instance.src_events:
        try:
            placeHolder += templateVocab[event]
        except:
            # placeHolder += np.array([-1] * 300)
            pass
    # placeHolder = placeHolder / len(instance.src_events)
    instance.setSimpleRepr(placeHolder)


def not_empty(s):
    return s and s.strip()


def like_camel_to_tokens(camel_format):
    simple_format = []
    temp = ''
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == '-' or camel_format[i] == '_':
                simple_format.append(temp)
                temp = ''
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ''
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ''
                temp += camel_format[i].lower()
                flag = True  # 需要回退
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(not_empty, simple_format))
    return simple_format


def nlp_emb_mergeTemplateEmbeddings_HDFS(dir, templates, logger):
    '''
    To merge token embeddings trained by fastText into templates embeddings.
    :return: No return value, but write result to a new .vec file.
    '''
    nlp_emb_file = 'dataset/nlp-word.vec'
    if not os.path.exists(dir):
        os.makedirs(dir)
    templates_emb_file = os.path.join(dir, 'templates_HDFS.vec')
    if os.path.exists(templates_emb_file):
        with open(templates_emb_file, 'r', encoding='utf-8') as reader:
            templateVocab = {}
            line_num = 0
            for line in reader.readlines():
                if line_num == 0:
                    vocabSize, embedSize = [int(x) for x in line.strip().split()]
                else:
                    items = line.strip().split()
                    if len(items) != embedSize + 1: continue
                    template_word, template_embedding = items[0], np.asarray(items[1:], dtype=np.float)
                    templateVocab[template_word] = template_embedding
                line_num += 1
    else:
        wordVocab = {}
        with open(nlp_emb_file, 'r', encoding='utf-8') as reader:
            embedSize = 300
            for line in reader.readlines():
                line = line.strip()
                tokens = line.split()
                if len(tokens) == embedSize + 1:
                    word, embed = tokens[0], np.asarray(tokens[1:], dtype=np.float)
                    wordVocab[word] = embed
        template_tokens = {}
        for template in set(templates):
            # template = template.replace("<*>", "*")
            tokens = template.split('$$')
            template_tokens[template] = tokens

        idf = {}
        wordCounter = Counter()
        total = len(template_tokens)
        for token in template_tokens.values():
            words = set(token)
            for word in words:
                wordCounter[word] += 1

        for word, count in wordCounter.most_common():
            idf[word] = np.log(total / count)

        with open(os.path.join(dir, 'idf'), 'w', encoding='utf-8') as writer:
            for token, idfScore in idf.items():
                writer.write(' '.join([token, str(idfScore)]) + '\n')
        templateVocab = {}
        for template in templates:
            placeHolder = np.zeros(embedSize)
            if template == 'this_is_an_empty_event':  # exist in  bgl: event is ''
                templateVocab[template] = placeHolder
            elif template not in templateVocab:
                tokens = template_tokens[template]
                wordCounter = Counter(tokens)
                for token in tokens:
                    simple_words = like_camel_to_tokens(token)
                    emb = np.zeros(embedSize)
                    for simple_word in simple_words:
                        if simple_word in wordVocab.keys():  # 默认是从现成的词向量文件里找的词向量
                            emb += wordVocab[simple_word]
                        else:  # 如果不在词向量文件里，会置为OOV
                            emb += np.zeros(embedSize)
                    emb = emb / len(simple_words)
                    tf = wordCounter[token] / len(tokens)
                    if token in idf.keys():
                        idf_score = idf[token]
                    else:
                        idf_score = 1
                    placeHolder += tf * idf_score * emb

                templateVocab[template] = placeHolder

        with open(templates_emb_file, 'w', encoding='utf-8')as writer:
            writer.write(str(len(templateVocab)) + ' ' + str(embedSize) + "\n")
            for template, embed in templateVocab.items():
                embed = ' '.join([str(x) for x in embed.tolist()])
                writer.write(' '.join([template, embed]) + "\n")
    logger.info('Generating/Loading template vocab finished, %d templates get.' % len(templateVocab))
    return templateVocab


def nlp_emb_mergeTemplateEmbeddings_BGL(dir, templates, dataset, logger):
    '''
    To merge token embeddings trained by fastText into templates embeddings.
    :param logger:
    :return: No return value, but write result to a new .vec file.
    '''
    nlp_emb_file = 'dataset/nlp-word.vec'
    vec_file = ''
    if not os.path.exists(dir):
        os.makedirs(dir)
    if dataset == 'bgl':
        vec_file = 'templates_BGL'
    elif dataset == 'tdb':
        vec_file = 'templates_Thunderbird'
    elif dataset == 'spirit':
        vec_file = 'templates_Spirit'
    elif dataset == "hdfs":
        vec_file = "templates_HDFS"
    else:
        logger.info("Unknown dataset")
        exit(-2)
    vec_file += '.vec'
    templates_emb_file = os.path.join(dir, vec_file)
    idf_file = os.path.join(dir, 'idf')
    if os.path.exists(templates_emb_file):
        with open(templates_emb_file, 'r', encoding='utf-8') as reader:
            templateVocab = {}
            line_num = 0
            for line in reader.readlines():
                if line_num == 0:
                    vocabSize, embedSize = [int(x) for x in line.strip().split()]
                else:
                    items = line.strip().split()
                    if len(items) != embedSize + 1: continue
                    template_word, template_embedding = items[0], np.asarray(items[1:], dtype=np.float)
                    templateVocab[template_word] = template_embedding
                line_num += 1
    else:
        wordVocab = {}
        with open(nlp_emb_file, 'r', encoding='utf-8') as reader:
            embedSize = 300
            for line in reader.readlines():
                line = line.strip()
                tokens = line.split()
                if len(tokens) == embedSize + 1:
                    word, embed = tokens[0], np.asarray(tokens[1:], dtype=np.float)
                    wordVocab[word] = embed

        pure_template_tokens = {}
        templates.append('$$'.join("No log during this period of time".strip().split()))
        for template in set(templates):
            line = ' '.join(template.split('$$'))
            tokens_simp = re.split(r'[,\!:=\[\]\(\)\$\s\.\/\#\|\\]', line.strip())  # 用以上符号分割
            # 去掉单独的 - _
            for index in range(len(tokens_simp)):
                if not re.match('[\_]+', tokens_simp[index]) is None:
                    tokens_simp[index] = ''
                if not re.match('[\-]+', tokens_simp[index]) is None:
                    tokens_simp[index] = ''
            tokens = list(filter(not_empty, tokens_simp))  # 除去 ''
            pure_template_tokens[template] = tokens

        # idf
        idf = {}
        wordCounter = Counter()
        total = len(pure_template_tokens)
        for tokens in pure_template_tokens.values():
            words = set(tokens)
            for word in words:
                wordCounter[word] += 1

        for word, count in wordCounter.most_common():
            idf[word] = np.log(total / count)

        with open(idf_file, 'w', encoding='utf-8') as writer:
            for token, idfScore in idf.items():
                writer.write(' '.join([token, str(idfScore)]) + '\n')

        templateVocab = {}
        for i, template in enumerate(templates):
            placeHolder = np.zeros(embedSize)
            if template == 'this_is_an_empty_event':  # exist: event is ''
                templateVocab[template] = placeHolder
            elif template not in templateVocab:
                tokens = pure_template_tokens[template]
                wordCounter = Counter(tokens)
                for token in tokens:
                    simple_words = like_camel_to_tokens(token)
                    emb = np.zeros(embedSize)
                    for simple_word in simple_words:
                        if simple_word in wordVocab.keys():  # 默认是从现成的词向量文件里找的词向量
                            emb += wordVocab[simple_word]
                        else:  # 如果不在词向量文件里，会置为OOV
                            emb += np.zeros(embedSize)
                    emb = emb / len(simple_words)
                    tf = wordCounter[token] / len(tokens)
                    if token in idf.keys():
                        idf_score = idf[token]
                    else:
                        idf_score = 1
                    placeHolder += tf * idf_score * emb

                templateVocab[template] = placeHolder

        with open(templates_emb_file, 'w', encoding='utf-8')as writer:
            writer.write(str(len(templateVocab)) + ' ' + str(embedSize) + "\n")
            for template, embed in templateVocab.items():
                embed = ' '.join([str(x) for x in embed.tolist()])
                writer.write(' '.join([template, embed]) + "\n")
    logger.info('Generating/Loading template vocab finished, %d templates get.' % len(templateVocab))
    return templateVocab
