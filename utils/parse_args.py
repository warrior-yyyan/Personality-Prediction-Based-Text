import argparse


def parse_args_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='essays')
    ap.add_argument('--token_len', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--embed_model', type=str, default='deberta-v3-large')
    ap.add_argument('--out_dir', type=str, default='/output/')
    ap.add_argument('--text_mode', type=str, default='256_head_tail')
    ap.add_argument('--embed_mode', type=str, default='mean')
    ap.add_argument('--local_model_path', type=str, default='/model/yyyan/deberta-v3-large')
    ap.add_argument('--local_tokenizer_path', type=str, default='/model/yyyan/deberta-v3-large/')
    args = ap.parse_args()
    return (
        args.dataset,
        args.token_len,
        args.batch_size,
        args.embed_model,
        args.out_dir,
        args.text_mode,
        args.embed_mode,
        args.local_model_path,
        args.local_tokenizer_path,
    )


def parse_args_classifier():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp_dir', type=str, default='features/')
    ap.add_argument('--dataset', type=str, default='essays')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--embed_model', type=str, default='deberta-v3-large')
    ap.add_argument('--n_layer', type=str, default='21')   # best layer 13 ~ 14 ~ 20 < 21 ~ 22
    ap.add_argument('--text_mode', type=str, default='256_head_tail')
    ap.add_argument('--embed_mode', type=str, default='mean')
    ap.add_argument('--ft_model', type=str, default='MLP')
    ap.add_argument('--jobid', type=int, default=0)
    args = ap.parse_args()
    return (
        args.inp_dir,
        args.dataset,
        args.lr,
        args.batch_size,
        args.epochs,
        args.embed_model,
        args.n_layer,
        args.text_mode,
        args.embed_mode,
        args.ft_model,
        args.jobid,
    )
