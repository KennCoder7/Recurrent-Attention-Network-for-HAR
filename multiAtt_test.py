from kenn.AttCaptionsModel import AttCaptionsModel
from kenn.AttCaptionsSolver import AttCaptionsSolver
from kenn.utils import load_captions_data


def main():
    # load train dataset
    data = load_captions_data(path='Data/CapData_new1/', split='test/')
    word_to_idx = data['word_to_idx']

    model = AttCaptionsModel(word_to_idx, dim_feature=[81, 128], dim_embed=6,
                             dim_hidden=128, n_time_step=3, prev2out=False,
                             ctx2out=True, alpha_c=1, selector=False, dropout=True)

    solver = AttCaptionsSolver(model, data,
                               pretrained_model=None, test_model='model/20190703-1/model-100',
                               log_path='log/', bool_val=True, bool_selector=False,
                               generated_caption_len=3)

    solver.test()


if __name__ == "__main__":
    main()
