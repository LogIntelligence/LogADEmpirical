from logadempirical.logparser import Spell, Drain


def parse_log(data_dir, output_dir, log_file, parser_type,log_format, regex, keep_para=False, st=0.3, depth=3, max_child=1000, tau=0.35):
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        # Drain is modified
        parser = Drain.LogParser(log_format,
                                 indir=data_dir,
                                 outdir=output_dir,
                                 depth=depth,
                                 st=st,
                                 rex=regex,
                                 keep_para=keep_para, maxChild=max_child)
        parser.parse(log_file)

    elif parser_type == "spell":
        # tau = 0.35
        parser = Spell.LogParser(indir=data_dir,
                                 outdir=output_dir,
                                 log_format=log_format,
                                 tau=tau,
                                 rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)
