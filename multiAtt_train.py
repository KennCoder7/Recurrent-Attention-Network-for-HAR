from kenn.AttCaptionsModel import AttCaptionsModel
from kenn.AttCaptionsSolver import AttCaptionsSolver
from kenn.utils import load_captions_data


def main():
    # load train dataset
    data = load_captions_data(path='Data/CapData_new1/', split='train/')
    word_to_idx = data['word_to_idx']

    model = AttCaptionsModel(word_to_idx, dim_feature=[81, 128], dim_embed=6,
                             dim_hidden=128, n_time_step=3, prev2out=False,
                             ctx2out=True, alpha_c=1, selector=False, dropout=True)

    solver = AttCaptionsSolver(model, data, n_epochs=100, batch_size=128, update_rule='adam',
                               learning_rate=0.0025, print_every=10, bool_save_model=True,
                               pretrained_model=None, model_path='model/20190703-1/',
                               log_path='log/', bool_val=True, generated_caption_len=3)

    solver.train()


if __name__ == "__main__":
    main()
