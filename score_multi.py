import argparse
import sys
import numpy as np
from collections import Counter

from fairseq import bleu, tokenizer
from fairseq.data import dictionary

dict = dictionary.Dictionary()
scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())


def dictolist(d):
    a = []
    for k in sorted(d.keys()):
        a.append(d[k])
    return a


def load_sys(path):
    with open(path) as f:
        lines = f.readlines()

    src, tgt, hypos, log_probs = {}, {}, {}, {}
    for line in lines:
        if line.startswith(('S-', 'T-', 'H-')):
            i = int(line[line.find('-')+1:line.find('\t')])
            if line.startswith('S-'):
                src[i] = line.split('\t')[1]
            if line.startswith('T-'):
                tgt[i] = line.split('\t')[1]
            if line.startswith('H-'):
                if i not in hypos:
                    hypos[i] = []
                    log_probs[i] = []
                hypos[i].append(line.split('\t')[2])
                log_probs[i].append(float(line.split('\t')[1]))

    return dictolist(src), dictolist(tgt), dictolist(hypos), dictolist(log_probs)


def load_ref(path):
    with open(path) as f:
        lines = f.readlines()

    src, tgt, refs = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith('S-'):
            src.append(lines[i].split('\t')[1])
            i += 1
        elif lines[i].startswith('T-'):
            tgt.append(lines[i].split('\t')[1])
            i += 1
        else:
            a = []
            while i < len(lines) and lines[i].startswith('R'):
                a.append(lines[i].split('\t')[1])
                i += 1
            refs.append(a)

    return src, tgt, refs


def merge(src, tgt, hypos, log_probs, path):
    with open(path, 'w') as f:
        for s, t, hs, lps in zip(src, tgt, hypos, log_probs):
            f.write(s)
            f.write(t)
            f.write('\n')
            for h, lp in zip(hs, lps):
                f.write('%f\t' % lp + h)
            f.write('------------------------------------------------------\n')


def corpus_bleu(ref, hypo):
    scorer.reset()
    for r, h in zip(ref, hypo):
        r_tok = tokenizer.Tokenizer.tokenize(r, dict)
        h_tok = tokenizer.Tokenizer.tokenize(h, dict)
        scorer.add(r_tok, h_tok)
    return scorer.score()


def sentence_bleu(ref, hypo):
    scorer.reset(one_init=True)
    r_tok = tokenizer.Tokenizer.tokenize(ref, dict)
    h_tok = tokenizer.Tokenizer.tokenize(hypo, dict)
    scorer.add(r_tok, h_tok)
    return scorer.score()


def pairwise(sents):
    _ref, _hypo = [], []
    for s in sents:
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    _ref.append(s[i])
                    _hypo.append(s[j])
    print('pairwise bleu score: %.2f' % corpus_bleu(_ref, _hypo))


def single_ref_oracle(ref, hypos):
    best = []
    cnt = Counter()
    for r, hs in zip(ref, hypos):
        s = [sentence_bleu(r, h) for h in hs]
        j = np.argmax(s)
        best.append(hs[j])
        cnt[j] += 1
    print('oracle count:')
    print(cnt)
    print('oracle bleu score: %.2f' % corpus_bleu(ref, best))


def single_ref_avg(ref, hypos):
    _ref, _hypo = [], []
    for r, hs in zip(ref, hypos):
        for h in hs:
            _ref.append(r)
            _hypo.append(h)
    print('avg bleu score: %.2f' % corpus_bleu(_ref, _hypo))


def single_ref_mostlikely(ref, hypos, log_probs):
    most_likely = [h[np.argmax(logp)] for h, logp in zip(hypos, log_probs)]
    print('single ref: %.2f' % corpus_bleu(ref, most_likely))

    hypos_t = list(map(list, zip(*hypos)))
    s = [corpus_bleu(ref, h) for h in hypos_t]
    np.set_printoptions(precision=1)
    print(np.array(s))


def filter_by_logprob(hypos, log_probs, threshold):
    hypos_, log_probs_ = [], []
    for hs, lps in zip(hypos, log_probs):
        hs_, lps_ = [], []
        for h, lp in zip(hs, lps):
            if lp >= threshold:
                hs_.append(h)
                lps_.append(lp)
        hypos_.append(hs_)
        log_probs_.append(lps_)
    return hypos_, log_probs_


def multi_ref(refs, hypos):
    _ref, _hypo = [], []
    ref_cnt = 0
    for rs, hs in zip(refs, hypos):
        a = set()
        for h in hs:
            s = [sentence_bleu(r, h) for r in rs]
            j = np.argmax(s)
            a.add(j)
            _ref.append(rs[j])
            _hypo.append(h)
        ref_cnt += len(a)
    print('avg oracle bleu score: %.2f' % corpus_bleu(_ref, _hypo))
    print('#refs covered: %.2f' % (ref_cnt / len(refs)))


'''def multi_ref(refs, hypos, threshold=0):
    _ref, _hypo = [], []
    ref_cnt = 0
    bad_hypo_cnt = 0
    for rs, hs in zip(refs, hypos):
        a = set()
        for h in hs:
            s = [sentence_bleu(r, h) for r in rs]
            j = np.argmax(s)
            if s[j] >= threshold:
                a.add(j)
                _ref.append(rs[j])
                _hypo.append(h)
            else:
                bad_hypo_cnt += 1
        ref_cnt += len(a)
    print('avg oracle bleu score: %.2f' % corpus_bleu(_ref, _hypo))
    print('#refs covered with bleu score >= %d: %.2f' % (threshold, ref_cnt / len(refs)))
    print('#hypos with highest bleu score < %d: %.2f' % (threshold, bad_hypo_cnt / len(hypos)))'''


def intra_ref(refs):
    cnt = sum(len(set(rs)) for rs in refs)
    print('avg distinct references: %.2f' % (cnt / len(refs)))
    pairwise(refs)

    _ref, _hypo = [], []
    for rs in refs:
        for i, h in enumerate(rs):
            rest = rs[:i] + rs[i+1:]
            s = [sentence_bleu(r, h) for r in rest]
            j = np.argmax(s)
            _ref.append(rest[j])
            _hypo.append(h)
    print('leave-one-out oracle bleu score among references: %.2f' % corpus_bleu(_ref, _hypo))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('--sys', default='', metavar='FILE',
                        help='path to system output')
    parser.add_argument('--output', default='', metavar='FILE',
                        help='print outputs into a pretty format')
    parser.add_argument('--ref', default='', metavar='FILE',
                        help='path to references')
    parser.add_argument('--pairwise', action='store_true',
                        help='compute the pairwise bleu score among hypotheses')
    parser.add_argument('--single-ref', action='store_true',
                        help='compute the bleu score between the most likely hypothesis and the target')
    parser.add_argument('--intra-ref', action='store_true',
                        help='compute intra-reference statistics')
    parser.add_argument('--nref', default=None, type=int, metavar='N',
                        help='number of references considered, consider all references by default')
    parser.add_argument('--threshold-logprob', default=None, type=float, metavar='N',
                        help='filter hypotheses that have log prob less than the threshold')
    #parser.add_argument('--threshold-bleu', default=0, type=int, metavar='N',
    #                    help='matchings with bleu score less than the threshold do not count')

    args = parser.parse_args()

    if args.sys:
        src, tgt, hypos, log_probs = load_sys(args.sys)
        if args.output:
            merge(src, tgt, hypos, log_probs, args.output)
        if args.single_ref:
            single_ref_mostlikely(tgt, hypos, log_probs)
        if args.pairwise:
            pairwise(hypos)
    if args.ref:
        _, _, refs = load_ref(args.ref)
        if args.nref is not None:
            refs = [rs[:args.nref] for rs in refs]
        if args.intra_ref:
            intra_ref(refs)
    if args.sys and args.ref:
        if args.threshold_logprob:
            hypos, _ = filter_by_logprob(hypos, log_probs, args.threshold_logprob)
        multi_ref(refs, hypos)
